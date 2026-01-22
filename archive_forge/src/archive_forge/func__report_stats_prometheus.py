import functools
import typing as ty
import urllib
from urllib.parse import urlparse
import iso8601
import jmespath
from keystoneauth1 import adapter
from openstack import _log
from openstack import exceptions
from openstack import resource
def _report_stats_prometheus(self, response, url=None, method=None, exc=None):
    if response is not None and (not url):
        url = response.request.url
    if response is not None and (not method):
        method = response.request.method
    parsed_url = urlparse(url)
    endpoint = '{}://{}{}'.format(parsed_url.scheme, parsed_url.netloc, parsed_url.path)
    if response is not None:
        labels = dict(method=method, endpoint=endpoint, service_type=self.service_type, status_code=response.status_code)
        self._prometheus_counter.labels(**labels).inc()
        self._prometheus_histogram.labels(**labels).observe(response.elapsed.total_seconds() * 1000)