from taskflow import test
from taskflow.utils import misc
def test_unicode_text(self):
    data = u'привет'
    self._check(data, data)