import sys
from io import BytesIO
from stat import S_ISDIR
from ...bzr.bzrdir import BzrDirMetaFormat1
from ...bzr.serializer import format_registry as serializer_format_registry
from ...errors import IllegalPath
from ...repository import InterRepository, Repository
from ...tests import TestCase, TestCaseWithTransport
from ...transport import NoSuchFile
from . import xml4
from .bzrdir import BzrDirFormat6
from .repository import (InterWeaveRepo, RepositoryFormat4, RepositoryFormat5,
def test_shared_disk_layout(self):
    control = BzrDirMetaFormat1().initialize(self.get_url())
    repo = RepositoryFormat7().initialize(control, shared=True)
    t = control.get_repository_transport(None)
    with t.get('format') as f:
        self.assertEqualDiff(b'Bazaar-NG Repository format 7', f.read())
    with t.get('shared-storage') as f:
        self.assertEqualDiff(b'', f.read())
    self.assertTrue(S_ISDIR(t.stat('revision-store').st_mode))
    self.assertTrue(S_ISDIR(t.stat('weaves').st_mode))
    with t.get('inventory.weave') as f:
        self.assertEqualDiff(b'# bzr weave file v5\nw\nW\n', f.read())
    self.assertFalse(t.has('branch-lock'))