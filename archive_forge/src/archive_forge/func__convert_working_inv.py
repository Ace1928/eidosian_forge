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
def _convert_working_inv(self):
    inv = xml4.serializer_v4.read_inventory(self.branch._transport.get('inventory'))
    f = BytesIO()
    xml5.serializer_v5.write_inventory(inv, f, working=True)
    self.branch._transport.put_bytes('inventory', f.getvalue(), mode=self.controldir._get_file_mode())