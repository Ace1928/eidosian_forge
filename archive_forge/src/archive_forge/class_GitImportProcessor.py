import stat
from typing import Dict, Tuple
from fastimport import commands, parser, processor
from fastimport import errors as fastimport_errors
from .index import commit_tree
from .object_store import iter_tree_contents
from .objects import ZERO_SHA, Blob, Commit, Tag
class GitImportProcessor(processor.ImportProcessor):
    """An import processor that imports into a Git repository using Dulwich."""

    def __init__(self, repo, params=None, verbose=False, outf=None) -> None:
        processor.ImportProcessor.__init__(self, params, verbose)
        self.repo = repo
        self.last_commit = ZERO_SHA
        self.markers: Dict[bytes, bytes] = {}
        self._contents: Dict[bytes, Tuple[int, bytes]] = {}

    def lookup_object(self, objectish):
        if objectish.startswith(b':'):
            return self.markers[objectish[1:]]
        return objectish

    def import_stream(self, stream):
        p = parser.ImportParser(stream)
        self.process(p.iter_commands)
        return self.markers

    def blob_handler(self, cmd):
        """Process a BlobCommand."""
        blob = Blob.from_string(cmd.data)
        self.repo.object_store.add_object(blob)
        if cmd.mark:
            self.markers[cmd.mark] = blob.id

    def checkpoint_handler(self, cmd):
        """Process a CheckpointCommand."""

    def commit_handler(self, cmd):
        """Process a CommitCommand."""
        commit = Commit()
        if cmd.author is not None:
            author = cmd.author
        else:
            author = cmd.committer
        author_name, author_email, author_timestamp, author_timezone = author
        committer_name, committer_email, commit_timestamp, commit_timezone = cmd.committer
        commit.author = author_name + b' <' + author_email + b'>'
        commit.author_timezone = author_timezone
        commit.author_time = int(author_timestamp)
        commit.committer = committer_name + b' <' + committer_email + b'>'
        commit.commit_timezone = commit_timezone
        commit.commit_time = int(commit_timestamp)
        commit.message = cmd.message
        commit.parents = []
        if cmd.from_:
            cmd.from_ = self.lookup_object(cmd.from_)
            self._reset_base(cmd.from_)
        for filecmd in cmd.iter_files():
            if filecmd.name == b'filemodify':
                if filecmd.data is not None:
                    blob = Blob.from_string(filecmd.data)
                    self.repo.object_store.add(blob)
                    blob_id = blob.id
                else:
                    blob_id = self.lookup_object(filecmd.dataref)
                self._contents[filecmd.path] = (filecmd.mode, blob_id)
            elif filecmd.name == b'filedelete':
                del self._contents[filecmd.path]
            elif filecmd.name == b'filecopy':
                self._contents[filecmd.dest_path] = self._contents[filecmd.src_path]
            elif filecmd.name == b'filerename':
                self._contents[filecmd.new_path] = self._contents[filecmd.old_path]
                del self._contents[filecmd.old_path]
            elif filecmd.name == b'filedeleteall':
                self._contents = {}
            else:
                raise Exception('Command %s not supported' % filecmd.name)
        commit.tree = commit_tree(self.repo.object_store, ((path, hexsha, mode) for path, (mode, hexsha) in self._contents.items()))
        if self.last_commit != ZERO_SHA:
            commit.parents.append(self.last_commit)
        for merge in cmd.merges:
            commit.parents.append(self.lookup_object(merge))
        self.repo.object_store.add_object(commit)
        self.repo[cmd.ref] = commit.id
        self.last_commit = commit.id
        if cmd.mark:
            self.markers[cmd.mark] = commit.id

    def progress_handler(self, cmd):
        """Process a ProgressCommand."""

    def _reset_base(self, commit_id):
        if self.last_commit == commit_id:
            return
        self._contents = {}
        self.last_commit = commit_id
        if commit_id != ZERO_SHA:
            tree_id = self.repo[commit_id].tree
            for path, mode, hexsha in iter_tree_contents(self.repo.object_store, tree_id):
                self._contents[path] = (mode, hexsha)

    def reset_handler(self, cmd):
        """Process a ResetCommand."""
        if cmd.from_ is None:
            from_ = ZERO_SHA
        else:
            from_ = self.lookup_object(cmd.from_)
        self._reset_base(from_)
        self.repo.refs[cmd.ref] = from_

    def tag_handler(self, cmd):
        """Process a TagCommand."""
        tag = Tag()
        tag.tagger = cmd.tagger
        tag.message = cmd.message
        tag.name = cmd.tag
        self.repo.add_object(tag)
        self.repo.refs['refs/tags/' + tag.name] = tag.id

    def feature_handler(self, cmd):
        """Process a FeatureCommand."""
        raise fastimport_errors.UnknownFeature(cmd.feature_name)