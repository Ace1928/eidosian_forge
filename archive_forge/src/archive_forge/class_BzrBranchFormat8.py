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
class BzrBranchFormat8(BranchFormatMetadir):
    """Metadir format supporting storing locations of subtree branches."""

    def _branch_class(self):
        return BzrBranch8

    @classmethod
    def get_format_string(cls):
        """See BranchFormat.get_format_string()."""
        return b'Bazaar Branch Format 8 (needs bzr 1.15)\n'

    def get_format_description(self):
        """See BranchFormat.get_format_description()."""
        return 'Branch format 8'

    def initialize(self, a_controldir, name=None, repository=None, append_revisions_only=None):
        """Create a branch of this format in a_controldir."""
        utf8_files = [('last-revision', b'0 null:\n'), ('branch.conf', self._get_initial_config(append_revisions_only)), ('tags', b''), ('references', b'')]
        return self._initialize_helper(a_controldir, utf8_files, name, repository)

    def make_tags(self, branch):
        """See breezy.branch.BranchFormat.make_tags()."""
        return _mod_tag.BasicTags(branch)

    def supports_set_append_revisions_only(self):
        return True

    def supports_stacking(self):
        return True
    supports_reference_locations = True