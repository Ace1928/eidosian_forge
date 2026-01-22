import base64
import stat
from typing import Optional
import fastbencode as bencode
from .. import errors, foreign, trace, urlutils
from ..foreign import ForeignRevision, ForeignVcs, VcsMappingRegistry
from ..revision import NULL_REVISION, Revision
from .errors import NoPushSupport
from .hg import extract_hg_metadata, format_hg_metadata
from .roundtrip import (CommitSupplement, extract_bzr_metadata,
class BzrGitMapping(foreign.VcsMapping):
    """Class that maps between Git and Bazaar semantics."""
    experimental = False
    BZR_DUMMY_FILE = None

    def is_special_file(self, filename):
        return filename in (self.BZR_DUMMY_FILE,)

    def __init__(self):
        super().__init__(foreign_vcs_git)

    def __eq__(self, other):
        return type(self) == type(other) and self.revid_prefix == other.revid_prefix

    @classmethod
    def revision_id_foreign_to_bzr(cls, git_rev_id):
        """Convert a git revision id handle to a Bazaar revision id."""
        from dulwich.protocol import ZERO_SHA
        if git_rev_id == ZERO_SHA:
            return NULL_REVISION
        return b'%s:%s' % (cls.revid_prefix, git_rev_id)

    @classmethod
    def revision_id_bzr_to_foreign(cls, bzr_rev_id):
        """Convert a Bazaar revision id to a git revision id handle."""
        if not bzr_rev_id.startswith(b'%s:' % cls.revid_prefix):
            raise errors.InvalidRevisionId(bzr_rev_id, cls)
        return (bzr_rev_id[len(cls.revid_prefix) + 1:], cls())

    def generate_file_id(self, path):
        if isinstance(path, str):
            path = encode_git_path(path)
        if path == b'':
            return ROOT_ID
        return FILE_ID_PREFIX + escape_file_id(path)

    def parse_file_id(self, file_id):
        if file_id == ROOT_ID:
            return ''
        if not file_id.startswith(FILE_ID_PREFIX):
            raise ValueError
        return decode_git_path(unescape_file_id(file_id[len(FILE_ID_PREFIX):]))

    def import_unusual_file_modes(self, rev, unusual_file_modes):
        if unusual_file_modes:
            ret = [(path, unusual_file_modes[path]) for path in sorted(unusual_file_modes.keys())]
            rev.properties['file-modes'] = bencode.bencode(ret)

    def export_unusual_file_modes(self, rev):
        try:
            file_modes = rev.properties['file-modes']
        except KeyError:
            return {}
        else:
            return dict(bencode.bdecode(file_modes.encode('utf-8')))

    def _generate_git_svn_metadata(self, rev, encoding):
        try:
            git_svn_id = rev.properties['git-svn-id']
        except KeyError:
            return ''
        else:
            return '\ngit-svn-id: %s\n' % git_svn_id.encode(encoding)

    def _generate_hg_message_tail(self, rev):
        extra = {}
        renames = []
        branch = 'default'
        for name in rev.properties:
            if name == 'hg:extra:branch':
                branch = rev.properties['hg:extra:branch']
            elif name.startswith('hg:extra'):
                extra[name[len('hg:extra:'):]] = base64.b64decode(rev.properties[name])
            elif name == 'hg:renames':
                renames = bencode.bdecode(base64.b64decode(rev.properties['hg:renames']))
        ret = format_hg_metadata(renames, branch, extra)
        if not isinstance(ret, bytes):
            raise TypeError(ret)
        return ret

    def _extract_git_svn_metadata(self, rev, message):
        lines = message.split('\n')
        if not (lines[-1] == '' and len(lines) >= 2 and lines[-2].startswith('git-svn-id:')):
            return message
        git_svn_id = lines[-2].split(': ', 1)[1]
        rev.properties['git-svn-id'] = git_svn_id
        url, rev, uuid = parse_git_svn_id(git_svn_id)
        return '\n'.join(lines[:-2])

    def _extract_hg_metadata(self, rev, message):
        message, renames, branch, extra = extract_hg_metadata(message)
        if branch is not None:
            rev.properties['hg:extra:branch'] = branch
        for name, value in extra.items():
            rev.properties['hg:extra:' + name] = base64.b64encode(value)
        if renames:
            rev.properties['hg:renames'] = base64.b64encode(bencode.bencode([(new, old) for old, new in renames.items()]))
        return message

    def _extract_bzr_metadata(self, rev, message):
        message, metadata = extract_bzr_metadata(message)
        return (message, metadata)

    def _decode_commit_message(self, rev, message, encoding):
        if message is None:
            decoded_message = None
        else:
            decoded_message = message.decode(encoding)
        return (decoded_message, CommitSupplement())

    def _encode_commit_message(self, rev, message, encoding):
        if message is None:
            return None
        else:
            return message.encode(encoding)

    def export_commit(self, rev, tree_sha, parent_lookup, lossy, verifiers):
        """Turn a Bazaar revision in to a Git commit

        :param tree_sha: Tree sha for the commit
        :param parent_lookup: Function for looking up the GIT sha equiv of a
            bzr revision
        :param lossy: Whether to store roundtripping information.
        :param verifiers: Verifiers info
        :return dulwich.objects.Commit represent the revision:
        """
        from dulwich.objects import Commit, Tag
        commit = Commit()
        commit.tree = tree_sha
        if not lossy:
            metadata = CommitSupplement()
            metadata.verifiers = verifiers
        else:
            metadata = None
        parents = []
        for p in rev.parent_ids:
            try:
                git_p = parent_lookup(p)
            except KeyError:
                git_p = None
                if metadata is not None:
                    metadata.explicit_parent_ids = rev.parent_ids
            if git_p is not None:
                if len(git_p) != 40:
                    raise AssertionError('unexpected length for %r' % git_p)
                parents.append(git_p)
        commit.parents = parents
        try:
            encoding = rev.properties['git-explicit-encoding']
        except KeyError:
            encoding = rev.properties.get('git-implicit-encoding', 'utf-8')
        try:
            commit.encoding = rev.properties['git-explicit-encoding'].encode('ascii')
        except KeyError:
            pass
        commit.committer = fix_person_identifier(rev.committer.encode(encoding))
        first_author = rev.get_apparent_authors()[0]
        if ',' in first_author and first_author.count('>') > 1:
            first_author = first_author.split(',')[0]
        commit.author = fix_person_identifier(first_author.encode(encoding))
        long = getattr(__builtins__, 'long', int)
        commit.commit_time = long(rev.timestamp)
        if 'author-timestamp' in rev.properties:
            commit.author_time = long(rev.properties['author-timestamp'])
        else:
            commit.author_time = commit.commit_time
        commit._commit_timezone_neg_utc = 'commit-timezone-neg-utc' in rev.properties
        commit.commit_timezone = rev.timezone
        commit._author_timezone_neg_utc = 'author-timezone-neg-utc' in rev.properties
        if 'author-timezone' in rev.properties:
            commit.author_timezone = int(rev.properties['author-timezone'])
        else:
            commit.author_timezone = commit.commit_timezone
        if 'git-gpg-signature' in rev.properties:
            commit.gpgsig = rev.properties['git-gpg-signature'].encode('utf-8', 'surrogateescape')
        commit.message = self._encode_commit_message(rev, rev.message, encoding)
        if not isinstance(commit.message, bytes):
            raise TypeError(commit.message)
        if metadata is not None:
            try:
                mapping_registry.parse_revision_id(rev.revision_id)
            except errors.InvalidRevisionId:
                metadata.revision_id = rev.revision_id
            mapping_properties = {'author', 'author-timezone', 'author-timezone-neg-utc', 'commit-timezone-neg-utc', 'git-implicit-encoding', 'git-gpg-signature', 'git-explicit-encoding', 'author-timestamp', 'file-modes'}
            for k, v in rev.properties.items():
                if k not in mapping_properties:
                    metadata.properties[k] = v
        if not lossy and metadata:
            if self.roundtripping:
                commit.message = inject_bzr_metadata(commit.message, metadata, encoding)
            else:
                raise NoPushSupport(None, None, self, revision_id=rev.revision_id)
        if not isinstance(commit.message, bytes):
            raise TypeError(commit.message)
        i = 0
        propname = 'git-mergetag-0'
        while propname in rev.properties:
            commit.mergetag.append(Tag.from_string(rev.properties[propname].encode('utf-8', 'surrogateescape')))
            i += 1
            propname = 'git-mergetag-%d' % i
        try:
            extra = commit._extra
        except AttributeError:
            extra = commit.extra
        if 'git-extra' in rev.properties:
            for l in rev.properties['git-extra'].splitlines():
                k, v = l.split(' ', 1)
                extra.append((k.encode('utf-8', 'surrogateescape'), v.encode('utf-8', 'surrogateescape')))
        return commit

    def get_revision_id(self, commit):
        if commit.encoding:
            encoding = commit.encoding.decode('ascii')
        else:
            encoding = 'utf-8'
        if commit.message is not None:
            try:
                message, metadata = self._decode_commit_message(None, commit.message, encoding)
            except UnicodeDecodeError:
                pass
            else:
                if metadata.revision_id:
                    return metadata.revision_id
        return self.revision_id_foreign_to_bzr(commit.id)

    def import_commit(self, commit, lookup_parent_revid, strict=True):
        """Convert a git commit to a bzr revision.

        :return: a `breezy.revision.Revision` object, foreign revid and a
            testament sha1
        """
        if commit is None:
            raise AssertionError("Commit object can't be None")
        rev = ForeignRevision(commit.id, self, self.revision_id_foreign_to_bzr(commit.id))
        rev.git_metadata = None

        def decode_using_encoding(rev, commit, encoding):
            try:
                rev.committer = commit.committer.decode(encoding)
            except LookupError:
                raise UnknownCommitEncoding(encoding)
            try:
                if commit.committer != commit.author:
                    rev.properties['author'] = commit.author.decode(encoding)
            except LookupError:
                raise UnknownCommitEncoding(encoding)
            rev.message, rev.git_metadata = self._decode_commit_message(rev, commit.message, encoding)
        if commit.encoding is not None:
            rev.properties['git-explicit-encoding'] = commit.encoding.decode('ascii')
        if commit.encoding is not None and commit.encoding != b'false':
            decode_using_encoding(rev, commit, commit.encoding.decode('ascii'))
        else:
            for encoding in ('utf-8', 'latin1'):
                try:
                    decode_using_encoding(rev, commit, encoding)
                except UnicodeDecodeError:
                    pass
                else:
                    if encoding != 'utf-8':
                        rev.properties['git-implicit-encoding'] = encoding
                    break
        if commit.commit_time != commit.author_time:
            rev.properties['author-timestamp'] = str(commit.author_time)
        if commit.commit_timezone != commit.author_timezone:
            rev.properties['author-timezone'] = '%d' % commit.author_timezone
        if commit._author_timezone_neg_utc:
            rev.properties['author-timezone-neg-utc'] = ''
        if commit._commit_timezone_neg_utc:
            rev.properties['commit-timezone-neg-utc'] = ''
        if commit.gpgsig:
            rev.properties['git-gpg-signature'] = commit.gpgsig.decode('utf-8', 'surrogateescape')
        if commit.mergetag:
            for i, tag in enumerate(commit.mergetag):
                rev.properties['git-mergetag-%d' % i] = tag.as_raw_string().decode('utf-8', 'surrogateescape')
        rev.timestamp = commit.commit_time
        rev.timezone = commit.commit_timezone
        rev.parent_ids = None
        if rev.git_metadata is not None:
            md = rev.git_metadata
            roundtrip_revid = md.revision_id
            if md.explicit_parent_ids:
                rev.parent_ids = md.explicit_parent_ids
            rev.properties.update(md.properties)
            verifiers = md.verifiers
        else:
            roundtrip_revid = None
            verifiers = {}
        if rev.parent_ids is None:
            parents = []
            for p in commit.parents:
                try:
                    parents.append(lookup_parent_revid(p))
                except KeyError:
                    parents.append(self.revision_id_foreign_to_bzr(p))
            rev.parent_ids = list(parents)
        unknown_extra_fields = []
        extra_lines = []
        try:
            extra = commit._extra
        except AttributeError:
            extra = commit.extra
        for k, v in extra:
            if k == HG_RENAME_SOURCE:
                extra_lines.append(k.decode('utf-8', 'surrogateescape') + ' ' + v.decode('utf-8', 'surrogateescape') + '\n')
            elif k == HG_EXTRA:
                hgk, hgv = v.split(b':', 1)
                if hgk not in (HG_EXTRA_AMEND_SOURCE, HG_EXTRA_REBASE_SOURCE, HG_EXTRA_ABSORB_SOURCE, HG_EXTRA_INTERMEDIATE_SOURCE, HG_EXTRA_SOURCE, HG_EXTRA_TOPIC, HG_EXTRA_REWRITE_NOISE) and strict:
                    raise UnknownMercurialCommitExtra(commit, [hgk])
                extra_lines.append(k.decode('utf-8', 'surrogateescape') + ' ' + v.decode('utf-8', 'surrogateescape') + '\n')
            else:
                unknown_extra_fields.append(k)
        if unknown_extra_fields and strict:
            raise UnknownCommitExtra(commit, [f.decode('ascii', 'replace') for f in unknown_extra_fields])
        if extra_lines:
            rev.properties['git-extra'] = ''.join(extra_lines)
        return (rev, roundtrip_revid, verifiers)