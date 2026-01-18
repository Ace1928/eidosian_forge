import os
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..bundle import Bundle, read_bundle, write_bundle
from ..pack import PackData, write_pack_objects
def test_roundtrip_bundle(self):
    origbundle = Bundle()
    origbundle.version = 3
    origbundle.capabilities = {'foo': None}
    origbundle.references = {b'refs/heads/master': b'ab' * 20}
    origbundle.prerequisites = [(b'cc' * 20, 'comment')]
    b = BytesIO()
    write_pack_objects(b.write, [])
    b.seek(0)
    origbundle.pack_data = PackData.from_file(b)
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'foo'), 'wb') as f:
            write_bundle(f, origbundle)
        with open(os.path.join(td, 'foo'), 'rb') as f:
            newbundle = read_bundle(f)
            self.assertEqual(origbundle, newbundle)