import sys
import json
import functools
from datetime import datetime
from unittest import mock
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib, parse_qs, urlparse, urlunquote
from libcloud.common.types import LibcloudError
from libcloud.compute.base import NodeSize, NodeLocation, StorageVolume, VolumeSnapshot
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.azure_arm import (
class AzureMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('azure_arm')
    responses = []

    def _update(self, fixture, body):
        for key, value in body.items():
            if isinstance(value, dict):
                fixture[key] = self._update(fixture.get(key, {}), value)
            else:
                fixture[key] = body[key]
        return fixture

    def __getattr__(self, n):

        def fn(method, url, body, headers):
            file_name = n.replace('99999999_9999_9999_9999_999999999999', AzureNodeDriverTests.SUBSCRIPTION_ID)
            unquoted_url = urlunquote(url)
            if '$skiptoken=' in unquoted_url and self.type != 'PAGINATION_INFINITE_LOOP':
                parsed_url = urlparse.urlparse(unquoted_url)
                params = parse_qs(parsed_url.query)
                file_name += '_' + params['$skiptoken'][0].split('!')[0]
            fixture = self.fixtures.load(file_name + '.json')
            if method in ('POST', 'PUT'):
                try:
                    body = json.loads(body)
                    fixture_tmp = json.loads(fixture)
                    fixture_tmp = self._update(fixture_tmp, body)
                    fixture = json.dumps(fixture_tmp)
                except ValueError:
                    pass
            if not n.endswith('_oauth2_token') and len(self.responses) > 0:
                f = self.responses.pop(0)
                return f(fixture)
            else:
                return (httplib.OK, fixture, headers, httplib.responses[httplib.OK])
        return fn