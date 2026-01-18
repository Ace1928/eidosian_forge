import testtools
from testtools import matchers
from yaql.language import contexts
from yaql.language import specs
def test_store_data(self):
    context = contexts.Context()
    context['key'] = 123
    self.assertEqual(123, context['key'])