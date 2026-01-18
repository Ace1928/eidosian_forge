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
def test_uses_lockdir(self):
    """repo format 7 actually locks on lockdir"""
    base_url = self.get_url()
    control = BzrDirMetaFormat1().initialize(base_url)
    repo = RepositoryFormat7().initialize(control, shared=True)
    t = control.get_repository_transport(None)
    repo.lock_write()
    repo.unlock()
    del repo
    repo = Repository.open(base_url)
    repo.lock_write()
    self.assertTrue(t.has('lock/held/info'))
    repo.unlock()
    self.assertFalse(t.has('lock/held/info'))