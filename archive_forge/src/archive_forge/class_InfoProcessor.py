from __future__ import absolute_import
from .. import (
from ..helpers import (
import stat
class InfoProcessor(processor.ImportProcessor):
    """An import processor that dumps statistics about the input.

    No changes to the current repository are made.

    As well as providing useful information about an import
    stream before importing it, this processor is useful for
    benchmarking the speed at which data can be extracted from
    the source.
    """

    def __init__(self, params=None, verbose=0, outf=None):
        processor.ImportProcessor.__init__(self, params, verbose, outf=outf)

    def pre_process(self):
        self.cmd_counts = {}
        for cmd in commands.COMMAND_NAMES:
            self.cmd_counts[cmd] = 0
        self.file_cmd_counts = {}
        for fc in commands.FILE_COMMAND_NAMES:
            self.file_cmd_counts[fc] = 0
        self.parent_counts = {}
        self.max_parent_count = 0
        self.committers = set()
        self.separate_authors_found = False
        self.symlinks_found = False
        self.executables_found = False
        self.sha_blob_references = False
        self.lightweight_tags = 0
        self.blobs = {}
        for usage in ['new', 'used', 'unknown', 'unmarked']:
            self.blobs[usage] = set()
        self.blob_ref_counts = {}
        self.reftracker = reftracker.RefTracker()
        self.merges = {}
        self.rename_old_paths = {}
        self.copy_source_paths = {}

    def post_process(self):
        cmd_names = commands.COMMAND_NAMES
        fc_names = commands.FILE_COMMAND_NAMES
        self._dump_stats_group('Command counts', [(c.decode('utf-8'), self.cmd_counts[c]) for c in cmd_names], str)
        self._dump_stats_group('File command counts', [(c.decode('utf-8'), self.file_cmd_counts[c]) for c in fc_names], str)
        if self.cmd_counts[b'commit']:
            p_items = []
            for i in range(self.max_parent_count + 1):
                if i in self.parent_counts:
                    count = self.parent_counts[i]
                    p_items.append(('parents-%d' % i, count))
            merges_count = len(self.merges)
            p_items.append(('total revisions merged', merges_count))
            flags = {'separate authors found': self.separate_authors_found, 'executables': self.executables_found, 'symlinks': self.symlinks_found, 'blobs referenced by SHA': self.sha_blob_references}
            self._dump_stats_group('Parent counts', p_items, str)
            self._dump_stats_group('Commit analysis', sorted(flags.items()), _found)
            heads = invert_dictset(self.reftracker.heads)
            self._dump_stats_group('Head analysis', [(k.decode('utf-8'), ', '.join([m.decode('utf-8') for m in v])) for k, v in heads.items()], None, _iterable_as_config_list)
            self._dump_stats_group('Merges', self.merges.items(), None)
            if self.verbose >= 2:
                self._dump_stats_group('Rename old paths', self.rename_old_paths.items(), len, _iterable_as_config_list)
                self._dump_stats_group('Copy source paths', self.copy_source_paths.items(), len, _iterable_as_config_list)
        if self.cmd_counts[b'blob']:
            if self.verbose:
                del self.blobs['used']
            self._dump_stats_group('Blob usage tracking', [(k, set([v1.decode() for v1 in v])) for k, v in self.blobs.items()], len, _iterable_as_config_list)
        if self.blob_ref_counts:
            blobs_by_count = invert_dict(self.blob_ref_counts)
            blob_items = sorted(blobs_by_count.items())
            self._dump_stats_group('Blob reference counts', blob_items, len, _iterable_as_config_list)
        if self.cmd_counts[b'reset']:
            reset_stats = {'lightweight tags': self.lightweight_tags}
            self._dump_stats_group('Reset analysis', reset_stats.items())

    def _dump_stats_group(self, title, items, normal_formatter=None, verbose_formatter=None):
        """Dump a statistics group.

        In verbose mode, do so as a config file so
        that other processors can load the information if they want to.
        :param normal_formatter: the callable to apply to the value
          before displaying it in normal mode
        :param verbose_formatter: the callable to apply to the value
          before displaying it in verbose mode
        """
        if self.verbose:
            self.outf.write('[%s]\n' % (title,))
            for name, value in items:
                if verbose_formatter is not None:
                    value = verbose_formatter(value)
                if type(name) == str:
                    name = name.replace(' ', '-')
                self.outf.write('%s = %s\n' % (name, value))
            self.outf.write('\n')
        else:
            self.outf.write('%s:\n' % (title,))
            for name, value in items:
                if normal_formatter is not None:
                    value = normal_formatter(value)
                self.outf.write('\t%s\t%s\n' % (value, name))

    def progress_handler(self, cmd):
        """Process a ProgressCommand."""
        self.cmd_counts[cmd.name] += 1

    def blob_handler(self, cmd):
        """Process a BlobCommand."""
        self.cmd_counts[cmd.name] += 1
        if cmd.mark is None:
            self.blobs['unmarked'].add(cmd.id)
        else:
            self.blobs['new'].add(cmd.id)
            try:
                self.blobs['used'].remove(cmd.id)
            except KeyError:
                pass

    def checkpoint_handler(self, cmd):
        """Process a CheckpointCommand."""
        self.cmd_counts[cmd.name] += 1

    def commit_handler(self, cmd):
        """Process a CommitCommand."""
        self.cmd_counts[cmd.name] += 1
        self.committers.add(cmd.committer)
        if cmd.author is not None:
            self.separate_authors_found = True
        for fc in cmd.iter_files():
            self.file_cmd_counts[fc.name] += 1
            if isinstance(fc, commands.FileModifyCommand):
                if fc.mode & 73:
                    self.executables_found = True
                if stat.S_ISLNK(fc.mode):
                    self.symlinks_found = True
                if fc.dataref is not None:
                    if fc.dataref[0] == ':':
                        self._track_blob(fc.dataref)
                    else:
                        self.sha_blob_references = True
            elif isinstance(fc, commands.FileRenameCommand):
                self.rename_old_paths.setdefault(cmd.id, set()).add(fc.old_path)
            elif isinstance(fc, commands.FileCopyCommand):
                self.copy_source_paths.setdefault(cmd.id, set()).add(fc.src_path)
        parents = self.reftracker.track_heads(cmd)
        parent_count = len(parents)
        try:
            self.parent_counts[parent_count] += 1
        except KeyError:
            self.parent_counts[parent_count] = 1
            if parent_count > self.max_parent_count:
                self.max_parent_count = parent_count
        if cmd.merges:
            for merge in cmd.merges:
                if merge in self.merges:
                    self.merges[merge] += 1
                else:
                    self.merges[merge] = 1

    def reset_handler(self, cmd):
        """Process a ResetCommand."""
        self.cmd_counts[cmd.name] += 1
        if cmd.ref.startswith(b'refs/tags/'):
            self.lightweight_tags += 1
        elif cmd.from_ is not None:
            self.reftracker.track_heads_for_ref(cmd.ref, cmd.from_)

    def tag_handler(self, cmd):
        """Process a TagCommand."""
        self.cmd_counts[cmd.name] += 1

    def feature_handler(self, cmd):
        """Process a FeatureCommand."""
        self.cmd_counts[cmd.name] += 1
        feature = cmd.feature_name
        if feature not in commands.FEATURE_NAMES:
            self.warning('feature %s is not supported - parsing may fail' % (feature,))

    def _track_blob(self, mark):
        if mark in self.blob_ref_counts:
            self.blob_ref_counts[mark] += 1
            pass
        elif mark in self.blobs['used']:
            self.blob_ref_counts[mark] = 2
            self.blobs['used'].remove(mark)
        elif mark in self.blobs['new']:
            self.blobs['used'].add(mark)
            self.blobs['new'].remove(mark)
        else:
            self.blobs['unknown'].add(mark)