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
def _report_stats_influxdb(self, response, url=None, method=None, exc=None):
    if response is not None and (not url):
        url = response.request.url
    if response is not None and (not method):
        method = response.request.method
    tags = dict(method=method, name='_'.join([normalize_metric_name(f) for f in self._extract_name(url, self.service_type, self.session.get_project_id())]))
    fields = dict(attempted=1)
    if response is not None:
        fields['duration'] = int(response.elapsed.total_seconds() * 1000)
        tags['status_code'] = str(response.status_code)
        fields[str(response.status_code)] = 1
        fields['%s.%s' % (method, response.status_code)] = 1
        fields['status_code_val'] = response.status_code
    elif exc:
        fields['failed'] = 1
    if 'additional_metric_tags' in self._influxdb_config:
        tags.update(self._influxdb_config['additional_metric_tags'])
    measurement = self._influxdb_config.get('measurement', 'openstack_api') if self._influxdb_config else 'openstack_api'
    measurement = '%s.%s' % (measurement, self.service_type)
    data = [dict(measurement=measurement, tags=tags, fields=fields)]
    try:
        self._influxdb_client.write_points(data)
    except Exception:
        self.log.exception('Error writing statistics to InfluxDB')