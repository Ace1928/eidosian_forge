from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from .. import errors, lockable_files
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import urlutils
from ..branch import (Branch, BranchFormat, BranchWriteLockResult,
from ..controldir import ControlDir
from ..decorators import only_raises
from ..lock import LogicalLockResult, _RelockDebugMixin
from ..trace import mutter
from . import bzrdir, rio
from .repository import MetaDirRepository
class BranchReferenceFormat(BranchFormatMetadir):
    """Bzr branch reference format.

    Branch references are used in implementing checkouts, they
    act as an alias to the real branch which is at some other url.

    This format has:
     - A location file
     - a format string
    """

    @classmethod
    def get_format_string(cls):
        """See BranchFormat.get_format_string()."""
        return b'Bazaar-NG Branch Reference Format 1\n'

    def get_format_description(self):
        """See BranchFormat.get_format_description()."""
        return 'Checkout reference format 1'

    def get_reference(self, a_controldir, name=None):
        """See BranchFormat.get_reference()."""
        transport = a_controldir.get_branch_transport(None, name=name)
        url = urlutils.strip_segment_parameters(a_controldir.user_url)
        return urlutils.join(url, transport.get_bytes('location').decode('utf-8'))

    def _write_reference(self, a_controldir, transport, to_branch):
        to_url = to_branch.user_url
        transport.put_bytes('location', to_url.encode('utf-8'))

    def set_reference(self, a_controldir, name, to_branch):
        """See BranchFormat.set_reference()."""
        transport = a_controldir.get_branch_transport(None, name=name)
        self._write_reference(a_controldir, transport, to_branch)

    def initialize(self, a_controldir, name=None, target_branch=None, repository=None, append_revisions_only=None):
        """Create a branch of this format in a_controldir."""
        if target_branch is None:
            raise errors.UninitializableFormat(self)
        mutter('creating branch reference in %s', a_controldir.user_url)
        if a_controldir._format.fixed_components:
            raise errors.IncompatibleFormat(self, a_controldir._format)
        if name is None:
            name = a_controldir._get_selected_branch()
        branch_transport = a_controldir.get_branch_transport(self, name=name)
        self._write_reference(a_controldir, branch_transport, target_branch)
        branch_transport.put_bytes('format', self.as_string())
        branch = self.open(a_controldir, name, _found=True, possible_transports=[target_branch.controldir.root_transport])
        self._run_post_branch_init_hooks(a_controldir, name, branch)
        return branch

    def _make_reference_clone_function(format, a_branch):
        """Create a clone() routine for a branch dynamically."""

        def clone(to_bzrdir, revision_id=None, repository_policy=None, name=None, tag_selector=None):
            """See Branch.clone()."""
            return format.initialize(to_bzrdir, target_branch=a_branch, name=name)
        return clone

    def open(self, a_controldir, name=None, _found=False, location=None, possible_transports=None, ignore_fallbacks=False, found_repository=None):
        """Return the branch that the branch reference in a_controldir points at.

        :param a_controldir: A BzrDir that contains a branch.
        :param name: Name of colocated branch to open, if any
        :param _found: a private parameter, do not use it. It is used to
            indicate if format probing has already be done.
        :param ignore_fallbacks: when set, no fallback branches will be opened
            (if there are any).  Default is to open fallbacks.
        :param location: The location of the referenced branch.  If
            unspecified, this will be determined from the branch reference in
            a_controldir.
        :param possible_transports: An optional reusable transports list.
        """
        if name is None:
            name = a_controldir._get_selected_branch()
        if not _found:
            format = BranchFormatMetadir.find_format(a_controldir, name=name)
            if format.__class__ != self.__class__:
                raise AssertionError('wrong format %r found for %r' % (format, self))
        if location is None:
            location = self.get_reference(a_controldir, name)
        real_bzrdir = ControlDir.open(location, possible_transports=possible_transports)
        result = real_bzrdir.open_branch(ignore_fallbacks=ignore_fallbacks, possible_transports=possible_transports)
        result.clone = self._make_reference_clone_function(result)
        return result