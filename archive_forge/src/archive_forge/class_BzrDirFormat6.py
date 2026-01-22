from io import BytesIO
from ... import errors, lockable_files
from ...bzr.bzrdir import BzrDir, BzrDirFormat, BzrDirMetaFormat1
from ...controldir import (ControlDir, Converter, MustHaveWorkingTree,
from ...i18n import gettext
from ...lazy_import import lazy_import
from ...transport import NoSuchFile, get_transport, local
import os
from breezy import (
from breezy.bzr import (
from breezy.plugins.weave_fmt.store.versioned import VersionedFileStore
from breezy.transactions import WriteTransaction
from breezy.plugins.weave_fmt import xml4
class BzrDirFormat6(BzrDirFormatAllInOne):
    """Bzr control format 6.

    This format is a combined format for working tree, branch and repository.
    It has:
     - Format 2 working trees [always]
     - Format 4 branches [always]
     - Format 6 repositories [always]
    """
    _lock_class = lockable_files.TransportLock

    def __eq__(self, other):
        return isinstance(self, type(other))

    @classmethod
    def get_format_string(cls):
        """See BzrDirFormat.get_format_string()."""
        return b'Bazaar-NG branch, format 6\n'

    def get_format_description(self):
        """See ControlDirFormat.get_format_description()."""
        return 'All-in-one format 6'

    def get_branch_format(self):
        from .branch import BzrBranchFormat4
        return BzrBranchFormat4()

    def get_converter(self, format=None):
        """See ControlDirFormat.get_converter()."""
        return ConvertBzrDir6ToMeta()

    def _initialize_for_clone(self, url):
        return self.initialize_on_transport(get_transport(url), _cloning=True)

    def initialize_on_transport(self, transport, _cloning=False):
        """Format 6 dirs always have working tree, branch and repository.

        Except when they are being cloned.
        """
        from .branch import BzrBranchFormat4
        from .repository import RepositoryFormat6
        result = super().initialize_on_transport(transport)
        RepositoryFormat6().initialize(result, _internal=True)
        if not _cloning:
            branch = BzrBranchFormat4().initialize(result)
            result._init_workingtree()
        return result

    def network_name(self):
        return self.get_format_string()

    def _open(self, transport):
        """See BzrDirFormat._open."""
        return BzrDir6(transport, self)

    def __return_repository_format(self):
        """Circular import protection."""
        from .repository import RepositoryFormat6
        return RepositoryFormat6()
    repository_format = property(__return_repository_format)