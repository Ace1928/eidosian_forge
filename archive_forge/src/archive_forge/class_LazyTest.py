from oslotest import base as test_base
from oslo_i18n import _lazy
class LazyTest(test_base.BaseTestCase):

    def setUp(self):
        super(LazyTest, self).setUp()
        self._USE_LAZY = _lazy.USE_LAZY

    def tearDown(self):
        _lazy.USE_LAZY = self._USE_LAZY
        super(LazyTest, self).tearDown()

    def test_enable_lazy(self):
        _lazy.USE_LAZY = False
        _lazy.enable_lazy()
        self.assertTrue(_lazy.USE_LAZY)

    def test_disable_lazy(self):
        _lazy.USE_LAZY = True
        _lazy.enable_lazy(False)
        self.assertFalse(_lazy.USE_LAZY)