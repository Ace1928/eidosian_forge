from urllib import parse as parser
from oslo_config import cfg
from osprofiler.drivers import base
from osprofiler import exc
def _hits(self, response):
    """Returns all hits of search query using scrolling

        :param response: ElasticSearch query response
        """
    scroll_id = response['_scroll_id']
    scroll_size = len(response['hits']['hits'])
    result = []
    while scroll_size > 0:
        for hit in response['hits']['hits']:
            result.append(hit['_source'])
        response = self.client.scroll(scroll_id=scroll_id, scroll=self.conf.profiler.es_scroll_time)
        scroll_id = response['_scroll_id']
        scroll_size = len(response['hits']['hits'])
    return result