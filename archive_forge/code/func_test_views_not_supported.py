from ..views import FileOutsideView, NoSuchView, ViewsNotSupported
from . import TestCase
def test_views_not_supported(self):
    err = ViewsNotSupported('atree')
    err_str = str(err)
    self.assertStartsWith(err_str, 'Views are not supported by ')
    self.assertEndsWith(err_str, "; use 'brz upgrade' to change your tree to a later format.")