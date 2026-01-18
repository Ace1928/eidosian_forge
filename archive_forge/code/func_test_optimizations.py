from zope.interface._compat import _should_attempt_c_optimizations
def test_optimizations(self):
    used = self._getTargetClass()
    fallback = self._getFallbackClass()
    if _should_attempt_c_optimizations():
        self.assertIsNot(used, fallback)
    else:
        self.assertIs(used, fallback)