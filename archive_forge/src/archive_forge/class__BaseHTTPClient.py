import copy
import io
import logging
import socket
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as ksa_exc
import OpenSSL
from oslo_utils import importutils
from oslo_utils import netutils
import requests
import urllib.parse
from oslo_utils import encodeutils
from glanceclient.common import utils
from glanceclient import exc
class _BaseHTTPClient(object):

    @staticmethod
    def _chunk_body(body):
        chunk = body
        while chunk:
            chunk = body.read(CHUNKSIZE)
            if not chunk:
                break
            yield chunk

    def _set_common_request_kwargs(self, headers, kwargs):
        """Handle the common parameters used to send the request."""
        content_type = headers.get('Content-Type', 'application/octet-stream')
        data = kwargs.pop('data', None)
        if data is not None and (not isinstance(data, str)):
            try:
                data = json.dumps(data)
                content_type = 'application/json'
            except TypeError:
                data = self._chunk_body(data)
        headers['Content-Type'] = content_type
        kwargs['stream'] = content_type == 'application/octet-stream'
        return data

    def _handle_response(self, resp):
        if not resp.ok:
            LOG.debug('Request returned failure status %s.', resp.status_code)
            raise exc.from_response(resp, resp.content)
        elif resp.status_code == requests.codes.MULTIPLE_CHOICES and resp.request.path_url != '/versions':
            raise exc.from_response(resp)
        content_type = resp.headers.get('Content-Type')
        if content_type == 'application/octet-stream':
            body_iter = _close_after_stream(resp, CHUNKSIZE)
        else:
            content = resp.text
            if content_type and content_type.startswith('application/json'):
                body_iter = resp.json()
            else:
                body_iter = io.StringIO(content)
                try:
                    body_iter = json.loads(''.join([c for c in body_iter]))
                except ValueError:
                    body_iter = None
        return (resp, body_iter)