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
class TestSerializer(TestCase):
    """Test serializer"""

    def test_registry(self):
        self.assertIs(xml4.serializer_v4, serializer_format_registry.get('4'))

    def test_canned_inventory(self):
        """Test unpacked a canned inventory v4 file."""
        inp = BytesIO(_working_inventory_v4)
        inv = xml4.serializer_v4.read_inventory(inp)
        self.assertEqual(len(inv), 4)
        self.assertTrue(inv.has_id(b'bar-20050901064931-73b4b1138abc9cd2'))

    def test_unpack_revision(self):
        """Test unpacking a canned revision v4"""
        inp = BytesIO(_revision_v4)
        rev = xml4.serializer_v4.read_revision(inp)
        eq = self.assertEqual
        eq(rev.committer, 'Martin Pool <mbp@sourcefrog.net>')
        eq(rev.inventory_id, 'mbp@sourcefrog.net-20050905080035-e0439293f8b6b9f9')
        eq(len(rev.parent_ids), 1)
        eq(rev.parent_ids[0], 'mbp@sourcefrog.net-20050905063503-43948f59fa127d92')