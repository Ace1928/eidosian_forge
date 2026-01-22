from oslotest import base
from aodhclient import utils
def _do_test(self, expr, expected):
    req = utils.search_query_builder(expr)
    self.assertEqual(expected, req)