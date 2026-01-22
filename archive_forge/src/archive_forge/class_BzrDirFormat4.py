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
class BzrDirFormat4(BzrDirFormat):
    """Bzr dir format 4.

    This format is a combined format for working tree, branch and repository.
    It has:
     - Format 1 working trees [always]
     - Format 4 branches [always]
     - Format 4 repositories [always]

    This format is deprecated: it indexes texts using a text it which is
    removed in format 5; write support for this format has been removed.
    """
    _lock_class = lockable_files.TransportLock

    def __eq__(self, other):
        return isinstance(self, type(other))

    @classmethod
    def get_format_string(cls):
        """See BzrDirFormat.get_format_string()."""
        return b'Bazaar-NG branch, format 0.0.4\n'

    def get_format_description(self):
        """See ControlDirFormat.get_format_description()."""
        return 'All-in-one format 4'

    def get_converter(self, format=None):
        """See ControlDirFormat.get_converter()."""
        return ConvertBzrDir4To5()

    def initialize_on_transport(self, transport):
        """Format 4 branches cannot be created."""
        raise errors.UninitializableFormat(self)

    def is_supported(self):
        """Format 4 is not supported.

        It is not supported because the model changed from 4 to 5 and the
        conversion logic is expensive - so doing it on the fly was not
        feasible.
        """
        return False

    def network_name(self):
        return self.get_format_string()

    def _open(self, transport):
        """See BzrDirFormat._open."""
        return BzrDir4(transport, self)

    def __return_repository_format(self):
        """Circular import protection."""
        from .repository import RepositoryFormat4
        return RepositoryFormat4()
    repository_format = property(__return_repository_format)

    @classmethod
    def from_string(cls, format_string):
        if format_string != cls.get_format_string():
            raise AssertionError('unexpected format string %r' % format_string)
        return cls()