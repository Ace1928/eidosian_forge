import importlib
import logging
import os
import sys
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import session as ks_session
from oslo_utils import importutils
from barbicanclient import exceptions
class _HTTPClient(adapter.Adapter):

    def __init__(self, session, microversion, project_id=None, **kwargs):
        endpoint = kwargs.pop('endpoint', None)
        if endpoint:
            kwargs['endpoint_override'] = '{}/{}/'.format(endpoint.rstrip('/'), kwargs.get('version'))
        super().__init__(session, **kwargs)
        self.microversion = microversion
        if project_id is None:
            self._default_headers = dict()
        else:
            self._default_headers = {'X-Project-Id': project_id}

    def request(self, *args, **kwargs):
        headers = kwargs.setdefault('headers', {})
        headers.update(self._default_headers)
        kwargs.setdefault('raise_exc', False)
        resp = super(_HTTPClient, self).request(*args, **kwargs)
        self._check_status_code(resp)
        return resp

    def get(self, *args, **kwargs):
        headers = kwargs.setdefault('headers', {})
        headers.setdefault('Accept', 'application/json')
        return super(_HTTPClient, self).get(*args, **kwargs).json()

    def post(self, path, *args, **kwargs):
        path = self._fix_path(path)
        return super(_HTTPClient, self).post(path, *args, **kwargs).json()

    def _fix_path(self, path):
        if not path[-1] == '/':
            path += '/'
        return path

    def _get_raw(self, path, *args, **kwargs):
        return self.request(path, 'GET', *args, **kwargs).content

    def _check_status_code(self, resp):
        status = resp.status_code
        LOG.debug('Response status {0}'.format(status))
        if status == 401:
            LOG.error('Auth error: {0}'.format(self._get_error_message(resp)))
            raise exceptions.HTTPAuthError('{0}'.format(self._get_error_message(resp)))
        if not status or status >= 500:
            LOG.error('5xx Server error: {0}'.format(self._get_error_message(resp)))
            raise exceptions.HTTPServerError('{0}'.format(self._get_error_message(resp)), status)
        if status >= 400:
            LOG.error('4xx Client error: {0}'.format(self._get_error_message(resp)))
            raise exceptions.HTTPClientError('{0}'.format(self._get_error_message(resp)), status)

    def _get_error_message(self, resp):
        try:
            response_data = resp.json()
            message = response_data['title']
            description = response_data.get('description')
            if description:
                message = '{0}: {1}'.format(message, description)
        except ValueError:
            message = resp.content
        return message