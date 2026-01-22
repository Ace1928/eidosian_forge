import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
def assertIndexIsAB(self, index):
    self.assertEqual({(b'a',): ((b'c',),), (b'b',): ((b'a',),)}, index.get_parent_map(index.keys()))
    self.assertEqual(((b'a',), 1, 2), index.get_position((b'a',)))
    self.assertEqual(((b'b',), 2, 3), index.get_position((b'b',)))
    self.assertEqual([b'opt'], index.get_options((b'a',)))