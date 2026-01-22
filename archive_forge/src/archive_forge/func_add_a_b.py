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
def add_a_b(self, index, random_id=None):
    kwargs = {}
    if random_id is not None:
        kwargs['random_id'] = random_id
    index.add_records([((b'a',), [b'option'], ((b'a',), 0, 1), [(b'b',)]), ((b'a',), [b'opt'], ((b'a',), 1, 2), [(b'c',)]), ((b'b',), [b'option'], ((b'b',), 2, 3), [(b'a',)])], **kwargs)