from os.path import join, sep, dirname
from numpy.distutils.misc_util import (
from numpy.testing import (
class TestMinrelpath:

    def test_1(self):
        n = lambda path: path.replace('/', sep)
        assert_equal(minrelpath(n('aa/bb')), n('aa/bb'))
        assert_equal(minrelpath('..'), '..')
        assert_equal(minrelpath(n('aa/..')), '')
        assert_equal(minrelpath(n('aa/../bb')), 'bb')
        assert_equal(minrelpath(n('aa/bb/..')), 'aa')
        assert_equal(minrelpath(n('aa/bb/../..')), '')
        assert_equal(minrelpath(n('aa/bb/../cc/../dd')), n('aa/dd'))
        assert_equal(minrelpath(n('.././..')), n('../..'))
        assert_equal(minrelpath(n('aa/bb/.././../dd')), n('dd'))