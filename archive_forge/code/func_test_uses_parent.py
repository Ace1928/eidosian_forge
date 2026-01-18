from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_uses_parent(self):
    """Parent location is used as a basis by default"""
    errmsg = self.run_send([], rc=3, wd='grandparent')[1]
    self.assertContainsRe(errmsg, 'No submit branch known or specified')
    stdout, stderr = self.run_send([])
    self.assertEqual(stderr.count('Using saved parent location'), 1)
    self.assertBundleContains([b'rev3'], [])