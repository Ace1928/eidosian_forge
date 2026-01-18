from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
@ddt.data({'entity_types': [{'name': 'volume', 'version': '3.0', 'command': 'os-reset_status'}, {'name': 'backup', 'version': '3.0', 'command': 'os-reset_status'}, {'name': 'snapshot', 'version': '3.0', 'command': 'os-reset_status'}, {'name': None, 'version': '3.0', 'command': 'os-reset_status'}, {'name': 'group', 'version': '3.20', 'command': 'reset_status'}, {'name': 'group-snapshot', 'version': '3.19', 'command': 'reset_status'}], 'r_id': ['1234'], 'states': ['available', 'error', None]}, {'entity_types': [{'name': 'volume', 'version': '3.0', 'command': 'os-reset_status'}, {'name': 'backup', 'version': '3.0', 'command': 'os-reset_status'}, {'name': 'snapshot', 'version': '3.0', 'command': 'os-reset_status'}, {'name': None, 'version': '3.0', 'command': 'os-reset_status'}, {'name': 'group', 'version': '3.20', 'command': 'reset_status'}, {'name': 'group-snapshot', 'version': '3.19', 'command': 'reset_status'}], 'r_id': ['1234', '5678'], 'states': ['available', 'error', None]})
@ddt.unpack
def test_reset_state_normal(self, entity_types, r_id, states):
    for state in states:
        for t in entity_types:
            if state is None:
                expected = {t['command']: {}}
                cmd = '--os-volume-api-version %s reset-state %s' % (t['version'], ' '.join(r_id))
            else:
                expected = {t['command']: {'status': state}}
                cmd = '--os-volume-api-version %s reset-state --state %s %s' % (t['version'], state, ' '.join(r_id))
            if t['name'] is not None:
                cmd += ' --type %s' % t['name']
            self.run_command(cmd)
            name = t['name'] if t['name'] else 'volume'
            for re in r_id:
                self.assert_called_anytime('POST', '/%ss/%s/action' % (name.replace('-', '_'), re), body=expected)