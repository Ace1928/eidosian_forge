import webob
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import strutils
import requests
def _json_request(self, creds_json):
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post('%s/v2.0/s3tokens' % self._request_uri, headers=headers, data=creds_json, verify=self._verify, timeout=CONF.s3_token.timeout)
    except requests.exceptions.RequestException as e:
        self._logger.info('HTTP connection exception: %s', e)
        resp = self._deny_request('InvalidURI')
        raise ServiceError(resp)
    if response.status_code < 200 or response.status_code >= 300:
        self._logger.debug('Keystone reply error: status=%s reason=%s', response.status_code, response.reason)
        resp = self._deny_request('AccessDenied')
        raise ServiceError(resp)
    return response