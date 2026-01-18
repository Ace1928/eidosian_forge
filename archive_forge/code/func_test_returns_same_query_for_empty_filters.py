from unittest import mock
from heat.db import filters as db_filters
from heat.tests import common
def test_returns_same_query_for_empty_filters(self):
    filters = {}
    db_filters.exact_filter(self.query, self.model, filters)
    self.assertEqual(0, self.query.call_count)