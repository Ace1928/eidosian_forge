import os
import tempfile
import breezy
from .. import errors, osutils, tests
from ..osutils import abspath, pathjoin, realpath, relpath
class MoreTests(tests.TestCaseWithTransport):

    def test_relpath(self):
        """test for branch path lookups

        breezy.osutils._relpath do a simple but subtle
        job: given a path (either relative to cwd or absolute), work out
        if it is inside a branch and return the path relative to the base.
        """
        dtmp = tempfile.mkdtemp()
        self.addCleanup(osutils.rmtree, dtmp)
        dtmp = realpath(dtmp)

        def rp(p):
            return relpath(dtmp, p)
        self.assertEqual('foo', rp(pathjoin(dtmp, 'foo')))
        self.assertEqual('', rp(dtmp))
        self.assertRaises(errors.PathNotChild, rp, '/etc')
        self.assertRaises(errors.PathNotChild, rp, dtmp.rstrip('\\/') + '2')
        self.assertRaises(errors.PathNotChild, rp, dtmp.rstrip('\\/') + '2/foo')
        os.chdir(dtmp)
        self.assertEqual('foo/bar/quux', rp('foo/bar/quux'))
        self.assertEqual('foo', rp('foo'))
        self.assertEqual('foo', rp('./foo'))
        self.assertEqual('foo', rp(abspath('foo')))
        self.assertRaises(errors.PathNotChild, rp, '../foo')