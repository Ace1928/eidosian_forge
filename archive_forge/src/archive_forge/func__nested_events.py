import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
def _nested_events(self):
    links = [{'rel': 'self'}, {'rel': 'resource'}, {'rel': 'stack'}, {'rel': 'root_stack'}]
    return [{'id': 'p_eventid1', 'event_time': '2014-01-06T16:14:00Z', 'stack_id': '1', 'resource_name': 'the_stack', 'resource_status': 'CREATE_IN_PROGRESS', 'resource_status_reason': 'Stack CREATE started', 'links': links}, {'id': 'n_eventid1', 'event_time': '2014-01-06T16:15:00Z', 'stack_id': '2', 'resource_name': 'nested_stack', 'resource_status': 'CREATE_IN_PROGRESS', 'resource_status_reason': 'Stack CREATE started', 'links': links}, {'id': 'n_eventid2', 'event_time': '2014-01-06T16:16:00Z', 'stack_id': '2', 'resource_name': 'nested_stack', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'Stack CREATE completed', 'links': links}, {'id': 'p_eventid2', 'event_time': '2014-01-06T16:17:00Z', 'stack_id': '1', 'resource_name': 'the_stack', 'resource_status': 'CREATE_COMPLETE', 'resource_status_reason': 'Stack CREATE completed', 'links': links}]