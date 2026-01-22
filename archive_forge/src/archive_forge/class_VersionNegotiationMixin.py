from distutils.version import StrictVersion
import functools
from http import client as http_client
import json
import logging
import re
import textwrap
import time
from urllib import parse as urlparse
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common.i18n import _
from ironicclient import exc
class VersionNegotiationMixin(object):

    def negotiate_version(self, conn, resp):
        """Negotiate the server version

        Assumption: Called after receiving a 406 error when doing a request.

        :param conn: A connection object
        :param resp: The response object from http request
        """

        def _query_server(conn):
            if self.os_ironic_api_version and (not isinstance(self.os_ironic_api_version, list)) and (self.os_ironic_api_version != 'latest'):
                base_version = '/v%s' % str(self.os_ironic_api_version).split('.')[0]
            else:
                base_version = API_VERSION
            resp = self._make_simple_request(conn, 'GET', base_version)
            if not resp.ok:
                raise exc.from_response(resp, method='GET', url=base_version)
            return resp
        version_overridden = False
        if resp and hasattr(resp, 'request') and hasattr(resp.request, 'headers'):
            orig_hdr = resp.request.headers
            req_api_ver = orig_hdr.get('X-OpenStack-Ironic-API-Version', self.os_ironic_api_version)
        else:
            req_api_ver = self.os_ironic_api_version
        if resp and req_api_ver != self.os_ironic_api_version and (self.api_version_select_state == 'negotiated'):
            requested_version = req_api_ver
            version_overridden = True
        else:
            requested_version = self.os_ironic_api_version
        if not resp:
            resp = _query_server(conn)
        if self.api_version_select_state not in API_VERSION_SELECTED_STATES:
            raise RuntimeError(_('Error: self.api_version_select_state should be one of the values in: "%(valid)s" but had the value: "%(value)s"') % {'valid': ', '.join(API_VERSION_SELECTED_STATES), 'value': self.api_version_select_state})
        min_ver, max_ver = self._parse_version_headers(resp)
        if not max_ver:
            LOG.debug('No version header in response, requesting from server')
            resp = _query_server(conn)
            min_ver, max_ver = self._parse_version_headers(resp)
        if StrictVersion(max_ver) > StrictVersion(LATEST_VERSION):
            LOG.debug('Remote API version %(max_ver)s is greater than the version supported by ironicclient. Maximum available version is %(client_ver)s', {'max_ver': max_ver, 'client_ver': LATEST_VERSION})
            max_ver = LATEST_VERSION
        if self.api_version_select_state == 'user' and (not self._must_negotiate_version()) or (self.api_version_select_state == 'negotiated' and version_overridden):
            raise exc.UnsupportedVersion(textwrap.fill(_('Requested API version %(req)s is not supported by the server, client, or the requested operation is not supported by the requested version. Supported version range is %(min)s to %(max)s') % {'req': requested_version, 'min': min_ver, 'max': max_ver}))
        if self.api_version_select_state == 'negotiated':
            raise exc.UnsupportedVersion(textwrap.fill(_("No API version was specified or the requested operation was not supported by the client's negotiated API version %(req)s.  Supported version range is: %(min)s to %(max)s") % {'req': requested_version, 'min': min_ver, 'max': max_ver}))
        if isinstance(requested_version, str):
            if requested_version == 'latest':
                negotiated_ver = max_ver
            else:
                negotiated_ver = str(min(StrictVersion(requested_version), StrictVersion(max_ver)))
        elif isinstance(requested_version, list):
            if 'latest' in requested_version:
                raise ValueError(textwrap.fill(_("The 'latest' API version can not be requested in a list of versions. Please explicitly request 'latest' or request only versions between %(min)s to %(max)s") % {'min': min_ver, 'max': max_ver}))
            versions = []
            for version in requested_version:
                if min_ver <= StrictVersion(version) <= max_ver:
                    versions.append(StrictVersion(version))
            if versions:
                negotiated_ver = str(max(versions))
            else:
                raise exc.UnsupportedVersion(textwrap.fill(_("Requested API version specified and the requested operation was not supported by the client's requested API version %(req)s.  Supported version range is: %(min)s to %(max)s") % {'req': requested_version, 'min': min_ver, 'max': max_ver}))
        else:
            raise ValueError(textwrap.fill(_("Requested API version %(req)s type is unsupported. Valid types are Strings such as '1.1', 'latest' or a list of string values representing API versions.") % {'req': requested_version}))
        if StrictVersion(negotiated_ver) < StrictVersion(min_ver):
            negotiated_ver = min_ver
        self.api_version_select_state = 'negotiated'
        self.os_ironic_api_version = negotiated_ver
        LOG.debug('Negotiated API version is %s', negotiated_ver)
        endpoint_override = getattr(self, 'endpoint_override', None)
        host, port = get_server(endpoint_override)
        filecache.save_data(host=host, port=port, data=negotiated_ver)
        return negotiated_ver

    def _generic_parse_version_headers(self, accessor_func):
        min_ver = accessor_func('X-OpenStack-Ironic-API-Minimum-Version', None)
        max_ver = accessor_func('X-OpenStack-Ironic-API-Maximum-Version', None)
        return (min_ver, max_ver)

    def _parse_version_headers(self, accessor_func):
        raise NotImplementedError()

    def _make_simple_request(self, conn, method, url):
        raise NotImplementedError()

    def _must_negotiate_version(self):
        return self.api_version_select_state == 'user' and (self.os_ironic_api_version == 'latest' or isinstance(self.os_ironic_api_version, list))