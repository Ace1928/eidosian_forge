import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_yaqlify_decorator_with_parameters(self):

    @yaqlization.yaqlize(yaqlize_attributes=True)
    class C(object):
        attr = 555
    self.assertEqual(555, self.eval('$.attr', C()))