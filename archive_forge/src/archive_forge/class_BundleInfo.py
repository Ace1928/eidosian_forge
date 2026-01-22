import base64
import os
import pprint
from io import BytesIO
from ... import cache_utf8, osutils, timestamp
from ...errors import BzrError, NoSuchId, TestamentMismatch
from ...osutils import pathjoin, sha_string, sha_strings
from ...revision import NULL_REVISION, Revision
from ...trace import mutter, warning
from ...tree import InterTree, Tree
from ..inventory import (Inventory, InventoryDirectory, InventoryFile,
from ..inventorytree import InventoryTree
from ..testament import StrictTestament
from ..xml5 import serializer_v5
from . import apply_bundle
class BundleInfo:
    """This contains the meta information. Stuff that allows you to
    recreate the revision or inventory XML.
    """

    def __init__(self, bundle_format=None):
        self.bundle_format = None
        self.committer = None
        self.date = None
        self.message = None
        self.revisions = []
        self.real_revisions = []
        self.timestamp = None
        self.timezone = None
        self._validated_revisions_against_repo = False

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def complete_info(self):
        """This makes sure that all information is properly
        split up, based on the assumptions that can be made
        when information is missing.
        """
        from breezy.timestamp import unpack_highres_date
        if not self.timestamp and self.date:
            self.timestamp, self.timezone = unpack_highres_date(self.date)
        self.real_revisions = []
        for rev in self.revisions:
            if rev.timestamp is None:
                if rev.date is not None:
                    rev.timestamp, rev.timezone = unpack_highres_date(rev.date)
                else:
                    rev.timestamp = self.timestamp
                    rev.timezone = self.timezone
            if rev.message is None and self.message:
                rev.message = self.message
            if rev.committer is None and self.committer:
                rev.committer = self.committer
            self.real_revisions.append(rev.as_revision())

    def get_base(self, revision):
        revision_info = self.get_revision_info(revision.revision_id)
        if revision_info.base_id is not None:
            return revision_info.base_id
        if len(revision.parent_ids) == 0:
            return NULL_REVISION
        else:
            return revision.parent_ids[-1]

    def _get_target(self):
        """Return the target revision."""
        if len(self.real_revisions) > 0:
            return self.real_revisions[0].revision_id
        elif len(self.revisions) > 0:
            return self.revisions[0].revision_id
        return None
    target = property(_get_target, doc='The target revision id')

    def get_revision(self, revision_id):
        for r in self.real_revisions:
            if r.revision_id == revision_id:
                return r
        raise KeyError(revision_id)

    def get_revision_info(self, revision_id):
        for r in self.revisions:
            if r.revision_id == revision_id:
                return r
        raise KeyError(revision_id)

    def revision_tree(self, repository, revision_id, base=None):
        revision = self.get_revision(revision_id)
        base = self.get_base(revision)
        if base == revision_id:
            raise AssertionError()
        if not self._validated_revisions_against_repo:
            self._validate_references_from_repository(repository)
        revision_info = self.get_revision_info(revision_id)
        inventory_revision_id = revision_id
        bundle_tree = BundleTree(repository.revision_tree(base), inventory_revision_id)
        self._update_tree(bundle_tree, revision_id)
        inv = bundle_tree.inventory
        self._validate_inventory(inv, revision_id)
        self._validate_revision(bundle_tree, revision_id)
        return bundle_tree

    def _validate_references_from_repository(self, repository):
        """Now that we have a repository which should have some of the
        revisions we care about, go through and validate all of them
        that we can.
        """
        rev_to_sha = {}
        inv_to_sha = {}

        def add_sha(d, revision_id, sha1):
            if revision_id is None:
                if sha1 is not None:
                    raise BzrError('A Null revision should alwayshave a null sha1 hash')
                return
            if revision_id in d:
                if sha1 != d[revision_id]:
                    raise BzrError('** Revision %r referenced with 2 different sha hashes %s != %s' % (revision_id, sha1, d[revision_id]))
            else:
                d[revision_id] = sha1
        checked = {}
        for rev_info in self.revisions:
            checked[rev_info.revision_id] = True
            add_sha(rev_to_sha, rev_info.revision_id, rev_info.sha1)
        for rev, rev_info in zip(self.real_revisions, self.revisions):
            add_sha(inv_to_sha, rev_info.revision_id, rev_info.inventory_sha1)
        count = 0
        missing = {}
        for revision_id, sha1 in rev_to_sha.items():
            if repository.has_revision(revision_id):
                testament = StrictTestament.from_revision(repository, revision_id)
                local_sha1 = self._testament_sha1_from_revision(repository, revision_id)
                if sha1 != local_sha1:
                    raise BzrError('sha1 mismatch. For revision id {%s}local: %s, bundle: %s' % (revision_id, local_sha1, sha1))
                else:
                    count += 1
            elif revision_id not in checked:
                missing[revision_id] = sha1
        if len(missing) > 0:
            warning('Not all revision hashes could be validated. Unable validate %d hashes' % len(missing))
        mutter('Verified %d sha hashes for the bundle.' % count)
        self._validated_revisions_against_repo = True

    def _validate_inventory(self, inv, revision_id):
        """At this point we should have generated the BundleTree,
        so build up an inventory, and make sure the hashes match.
        """
        cs = serializer_v5.write_inventory_to_chunks(inv)
        sha1 = sha_strings(cs)
        rev = self.get_revision(revision_id)
        if rev.revision_id != revision_id:
            raise AssertionError()
        if sha1 != rev.inventory_sha1:
            with open(',,bogus-inv', 'wb') as f:
                f.writelines(cs)
            warning('Inventory sha hash mismatch for revision %s. %s != %s' % (revision_id, sha1, rev.inventory_sha1))

    def _testament(self, revision, tree):
        raise NotImplementedError(self._testament)

    def _validate_revision(self, tree, revision_id):
        """Make sure all revision entries match their checksum."""
        rev_to_sha1 = {}
        rev = self.get_revision(revision_id)
        rev_info = self.get_revision_info(revision_id)
        if not rev.revision_id == rev_info.revision_id:
            raise AssertionError()
        if not rev.revision_id == revision_id:
            raise AssertionError()
        testament = self._testament(rev, tree)
        sha1 = testament.as_sha1()
        if sha1 != rev_info.sha1:
            raise TestamentMismatch(rev.revision_id, rev_info.sha1, sha1)
        if rev.revision_id in rev_to_sha1:
            raise BzrError('Revision {%s} given twice in the list' % rev.revision_id)
        rev_to_sha1[rev.revision_id] = sha1

    def _update_tree(self, bundle_tree, revision_id):
        """This fills out a BundleTree based on the information
        that was read in.

        :param bundle_tree: A BundleTree to update with the new information.
        """

        def get_rev_id(last_changed, path, kind):
            if last_changed is not None:
                changed_revision_id = cache_utf8.encode(last_changed)
            else:
                changed_revision_id = revision_id
            bundle_tree.note_last_changed(path, changed_revision_id)
            return changed_revision_id

        def extra_info(info, new_path):
            last_changed = None
            encoding = None
            for info_item in info:
                try:
                    name, value = info_item.split(':', 1)
                except ValueError:
                    raise ValueError('Value %r has no colon' % info_item)
                if name == 'last-changed':
                    last_changed = value
                elif name == 'executable':
                    val = value == 'yes'
                    bundle_tree.note_executable(new_path, val)
                elif name == 'target':
                    bundle_tree.note_target(new_path, value)
                elif name == 'encoding':
                    encoding = value
            return (last_changed, encoding)

        def do_patch(path, lines, encoding):
            if encoding == 'base64':
                patch = base64.b64decode(b''.join(lines))
            elif encoding is None:
                patch = b''.join(lines)
            else:
                raise ValueError(encoding)
            bundle_tree.note_patch(path, patch)

        def renamed(kind, extra, lines):
            info = extra.split(' // ')
            if len(info) < 2:
                raise BzrError('renamed action lines need both a from and to: %r' % extra)
            old_path = info[0]
            if info[1].startswith('=> '):
                new_path = info[1][3:]
            else:
                new_path = info[1]
            bundle_tree.note_rename(old_path, new_path)
            last_modified, encoding = extra_info(info[2:], new_path)
            revision = get_rev_id(last_modified, new_path, kind)
            if lines:
                do_patch(new_path, lines, encoding)

        def removed(kind, extra, lines):
            info = extra.split(' // ')
            if len(info) > 1:
                raise BzrError('removed action lines should only have the path: %r' % extra)
            path = info[0]
            bundle_tree.note_deletion(path)

        def added(kind, extra, lines):
            info = extra.split(' // ')
            if len(info) <= 1:
                raise BzrError('add action lines require the path and file id: %r' % extra)
            elif len(info) > 5:
                raise BzrError('add action lines have fewer than 5 entries.: %r' % extra)
            path = info[0]
            if not info[1].startswith('file-id:'):
                raise BzrError('The file-id should follow the path for an add: %r' % extra)
            file_id = cache_utf8.encode(info[1][8:])
            bundle_tree.note_id(file_id, path, kind)
            bundle_tree.note_executable(path, False)
            last_changed, encoding = extra_info(info[2:], path)
            revision = get_rev_id(last_changed, path, kind)
            if kind == 'directory':
                return
            do_patch(path, lines, encoding)

        def modified(kind, extra, lines):
            info = extra.split(' // ')
            if len(info) < 1:
                raise BzrError('modified action lines have at leastthe path in them: %r' % extra)
            path = info[0]
            last_modified, encoding = extra_info(info[1:], path)
            revision = get_rev_id(last_modified, path, kind)
            if lines:
                do_patch(path, lines, encoding)
        valid_actions = {'renamed': renamed, 'removed': removed, 'added': added, 'modified': modified}
        for action_line, lines in self.get_revision_info(revision_id).tree_actions:
            first = action_line.find(' ')
            if first == -1:
                raise BzrError('Bogus action line (no opening space): %r' % action_line)
            second = action_line.find(' ', first + 1)
            if second == -1:
                raise BzrError('Bogus action line (missing second space): %r' % action_line)
            action = action_line[:first]
            kind = action_line[first + 1:second]
            if kind not in ('file', 'directory', 'symlink'):
                raise BzrError('Bogus action line (invalid object kind %r): %r' % (kind, action_line))
            extra = action_line[second + 1:]
            if action not in valid_actions:
                raise BzrError('Bogus action line (unrecognized action): %r' % action_line)
            valid_actions[action](kind, extra, lines)

    def install_revisions(self, target_repo, stream_input=True):
        """Install revisions and return the target revision

        :param target_repo: The repository to install into
        :param stream_input: Ignored by this implementation.
        """
        apply_bundle.install_bundle(target_repo, self)
        return self.target

    def get_merge_request(self, target_repo):
        """Provide data for performing a merge

        Returns suggested base, suggested target, and patch verification status
        """
        return (None, self.target, 'inapplicable')