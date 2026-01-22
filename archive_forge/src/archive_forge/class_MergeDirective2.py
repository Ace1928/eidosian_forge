import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
class MergeDirective2(BaseMergeDirective):
    _format_string = b'Bazaar merge directive format 2 (Bazaar 0.90)'

    def __init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch=None, source_branch=None, message=None, bundle=None, base_revision_id=None):
        if source_branch is None and bundle is None:
            raise errors.NoMergeSource()
        BaseMergeDirective.__init__(self, revision_id, testament_sha1, time, timezone, target_branch, patch, source_branch, message)
        self.bundle = bundle
        self.base_revision_id = base_revision_id

    def _patch_type(self):
        if self.bundle is not None:
            return 'bundle'
        elif self.patch is not None:
            return 'diff'
        else:
            return None
    patch_type = property(_patch_type)

    def clear_payload(self):
        self.patch = None
        self.bundle = None

    def get_raw_bundle(self):
        if self.bundle is None:
            return None
        else:
            return base64.b64decode(self.bundle)

    @classmethod
    def _from_lines(klass, line_iter):
        stanza = rio.read_patch_stanza(line_iter)
        patch = None
        bundle = None
        try:
            start = next(line_iter)
        except StopIteration:
            pass
        else:
            if start.startswith(b'# Begin patch'):
                patch_lines = []
                for line in line_iter:
                    if line.startswith(b'# Begin bundle'):
                        start = line
                        break
                    patch_lines.append(line)
                else:
                    start = None
                patch = b''.join(patch_lines)
            if start is not None:
                if start.startswith(b'# Begin bundle'):
                    bundle = b''.join(line_iter)
                else:
                    raise IllegalMergeDirectivePayload(start)
        time, timezone = timestamp.parse_patch_date(stanza.get('timestamp'))
        kwargs = {}
        for key in ('revision_id', 'testament_sha1', 'target_branch', 'source_branch', 'message', 'base_revision_id'):
            try:
                kwargs[key] = stanza.get(key)
            except KeyError:
                pass
        kwargs['revision_id'] = kwargs['revision_id'].encode('utf-8')
        kwargs['base_revision_id'] = kwargs['base_revision_id'].encode('utf-8')
        if 'testament_sha1' in kwargs:
            kwargs['testament_sha1'] = kwargs['testament_sha1'].encode('ascii')
        return klass(time=time, timezone=timezone, patch=patch, bundle=bundle, **kwargs)

    def to_lines(self):
        lines = self._to_lines(base_revision=True)
        if self.patch is not None:
            lines.append(b'# Begin patch\n')
            lines.extend(self.patch.splitlines(True))
        if self.bundle is not None:
            lines.append(b'# Begin bundle\n')
            lines.extend(self.bundle.splitlines(True))
        return lines

    @classmethod
    def from_objects(klass, repository, revision_id, time, timezone, target_branch, include_patch=True, include_bundle=True, local_target_branch=None, public_branch=None, message=None, base_revision_id=None):
        """Generate a merge directive from various objects

        :param repository: The repository containing the revision
        :param revision_id: The revision to merge
        :param time: The POSIX timestamp of the date the request was issued.
        :param timezone: The timezone of the request
        :param target_branch: The url of the branch to merge into
        :param include_patch: If true, include a preview patch
        :param include_bundle: If true, include a bundle
        :param local_target_branch: the target branch, either itself or a local copy
        :param public_branch: location of a public branch containing
            the target revision.
        :param message: Message to use when committing the merge
        :return: The merge directive

        The public branch is always used if supplied.  If no bundle is
        included, the public branch must be supplied, and will be verified.

        If the message is not supplied, the message from revision_id will be
        used for the commit.
        """
        with contextlib.ExitStack() as exit_stack:
            exit_stack.enter_context(repository.lock_write())
            t_revision_id = revision_id
            if revision_id == b'null:':
                t_revision_id = None
            t = testament.StrictTestament3.from_revision(repository, t_revision_id)
            if local_target_branch is None:
                submit_branch = _mod_branch.Branch.open(target_branch)
            else:
                submit_branch = local_target_branch
            exit_stack.enter_context(submit_branch.lock_read())
            if submit_branch.get_public_branch() is not None:
                target_branch = submit_branch.get_public_branch()
            submit_revision_id = submit_branch.last_revision()
            graph = repository.get_graph(submit_branch.repository)
            ancestor_id = graph.find_unique_lca(revision_id, submit_revision_id)
            if base_revision_id is None:
                base_revision_id = ancestor_id
            if (include_patch, include_bundle) != (False, False):
                repository.fetch(submit_branch.repository, submit_revision_id)
            if include_patch:
                patch = klass._generate_diff(repository, revision_id, base_revision_id)
            else:
                patch = None
            if include_bundle:
                bundle = base64.b64encode(klass._generate_bundle(repository, revision_id, ancestor_id))
            else:
                bundle = None
            if public_branch is not None and (not include_bundle):
                public_branch_obj = _mod_branch.Branch.open(public_branch)
                exit_stack.enter_context(public_branch_obj.lock_read())
                if not public_branch_obj.repository.has_revision(revision_id):
                    raise errors.PublicBranchOutOfDate(public_branch, revision_id)
            testament_sha1 = t.as_sha1()
        return klass(revision_id, testament_sha1, time, timezone, target_branch, patch, public_branch, message, bundle, base_revision_id)

    def _verify_patch(self, repository):
        calculated_patch = self._generate_diff(repository, self.revision_id, self.base_revision_id)
        stored_patch = re.sub(b'\r\n?', b'\n', self.patch)
        calculated_patch = re.sub(b'\r\n?', b'\n', calculated_patch)
        calculated_patch = re.sub(b' *\n', b'\n', calculated_patch)
        stored_patch = re.sub(b' *\n', b'\n', stored_patch)
        return calculated_patch == stored_patch

    def get_merge_request(self, repository):
        """Provide data for performing a merge

        Returns suggested base, suggested target, and patch verification status
        """
        verified = self._maybe_verify(repository)
        return (self.base_revision_id, self.revision_id, verified)

    def _maybe_verify(self, repository):
        if self.patch is not None:
            if self._verify_patch(repository):
                return 'verified'
            else:
                return 'failed'
        else:
            return 'inapplicable'