from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
class CloudSearchSearchFacetTest(CloudSearchSearchBaseTest):
    response = {'rank': '-text_relevance', 'match-expr': 'Test', 'hits': {'found': 30, 'start': 0, 'hit': CloudSearchSearchBaseTest.hits}, 'info': {'rid': 'b7c167f6c2da6d93531b9a7b314ad030b3a74803b4b7797edb905ba5a6a08', 'time-ms': 2, 'cpu-time-ms': 0}, 'facets': {'tags': {}, 'animals': {'constraints': [{'count': '2', 'value': 'fish'}, {'count': '1', 'value': 'lions'}]}}}

    def test_cloudsearch_search_facets(self):
        search = SearchConnection(endpoint=HOSTNAME)
        results = search.search(q='Test', facet=['tags'])
        self.assertTrue('tags' not in results.facets)
        self.assertEqual(results.facets['animals'], {u'lions': u'1', u'fish': u'2'})