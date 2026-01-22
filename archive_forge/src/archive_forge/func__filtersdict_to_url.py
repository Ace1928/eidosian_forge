from oslo_serialization import jsonutils
from aodhclient import utils
from aodhclient.v2 import alarm_cli
from aodhclient.v2 import base
@staticmethod
def _filtersdict_to_url(filters):
    urls = []
    for k, v in sorted(filters.items()):
        url = 'q.field=%s&q.op=eq&q.value=%s' % (k, v)
        urls.append(url)
    return '&'.join(urls)