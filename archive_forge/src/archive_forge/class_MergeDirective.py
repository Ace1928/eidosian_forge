import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
class MergeDirective(BaseMergeDirective):
    """A request to perform a merge into a branch.

    Designed to be serialized and mailed.  It provides all the information
    needed to perform a merge automatically, by providing at minimum a revision
    bundle or the location of a branch.

    The serialization format is robust against certain common forms of
    deterioration caused by mailing.

    The format is also designed to be patch-compatible.  If the directive
    includes a diff or revision bundle, it should be possible to apply it
    directly using the standard patch program.
    """
    _format_string = b'Bazaar merge directive format 1'

    def __init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, patch_type=None, source_branch=None, message=None, bundle=None):
        """Constructor.

        :param revision_id: The revision to merge
        :param testament_sha1: The sha1 of the testament of the revision to
            merge.
        :param time: The current POSIX timestamp time
        :param timezone: The timezone offset
        :param target_branch: Location of the branch to apply the merge to
        :param patch: The text of a diff or bundle
        :param patch_type: None, "diff" or "bundle", depending on the contents
            of patch
        :param source_branch: A public location to merge the revision from
        :param message: The message to use when committing this merge
        """
        BaseMergeDirective.__init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch, source_branch, message)
        if patch_type not in (None, 'diff', 'bundle'):
            raise ValueError(patch_type)
        if patch_type != 'bundle' and source_branch is None:
            raise errors.NoMergeSource()
        if patch_type is not None and patch is None:
            raise errors.PatchMissing(patch_type)
        self.patch_type = patch_type

    def clear_payload(self):
        self.patch = None
        self.patch_type = None

    def get_raw_bundle(self):
        return self.bundle

    def _bundle(self):
        if self.patch_type == 'bundle':
            return self.patch
        else:
            return None
    bundle = property(_bundle)

    @classmethod
    def from_lines(klass, lines):
        """Deserialize a MergeRequest from an iterable of lines

        :param lines: An iterable of lines
        :return: a MergeRequest
        """
        line_iter = iter(lines)
        firstline = b''
        for line in line_iter:
            if line.startswith(b'# Bazaar merge directive format '):
                return _format_registry.get(line[2:].rstrip())._from_lines(line_iter)
            firstline = firstline or line.strip()
        raise errors.NotAMergeDirective(firstline)

    @classmethod
    def _from_lines(klass, line_iter):
        stanza = rio.read_patch_stanza(line_iter)
        patch_lines = list(line_iter)
        if len(patch_lines) == 0:
            patch = None
            patch_type = None
        else:
            patch = b''.join(patch_lines)
            try:
                bundle_serializer.read_bundle(BytesIO(patch))
            except (errors.NotABundle, errors.BundleNotSupported, errors.BadBundle):
                patch_type = 'diff'
            else:
                patch_type = 'bundle'
        time, timezone = timestamp.parse_patch_date(stanza.get('timestamp'))
        kwargs = {}
        for key in ('revision_id', 'testament_sha1', 'target_branch', 'source_branch', 'message'):
            try:
                kwargs[key] = stanza.get(key)
            except KeyError:
                pass
        kwargs['revision_id'] = kwargs['revision_id'].encode('utf-8')
        if 'testament_sha1' in kwargs:
            kwargs['testament_sha1'] = kwargs['testament_sha1'].encode('ascii')
        return MergeDirective(time=time, timezone=timezone, patch_type=patch_type, patch=patch, **kwargs)

    def to_lines(self):
        lines = self._to_lines()
        if self.patch is not None:
            lines.extend(self.patch.splitlines(True))
        return lines

    @staticmethod
    def _generate_bundle(repository, revision_id, ancestor_id):
        s = BytesIO()
        bundle_serializer.write_bundle(repository, revision_id, ancestor_id, s, '0.9')
        return s.getvalue()

    def get_merge_request(self, repository):
        """Provide data for performing a merge

        Returns suggested base, suggested target, and patch verification status
        """
        return (None, self.revision_id, 'inapplicable')