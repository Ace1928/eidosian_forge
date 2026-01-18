import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import crypt
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import service
from heat.engine import service_software_config
from heat.engine import software_config_io as swc_io
from heat.objects import resource as resource_objects
from heat.objects import software_config as software_config_object
from heat.objects import software_deployment as software_deployment_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_update_software_deployment_new_config(self):
    server_id = str(uuid.uuid4())
    mock_push = self.patchobject(self.engine.software_config, '_push_metadata_software_deployments')
    deployment = self._create_software_deployment(server_id=server_id)
    self.assertIsNotNone(deployment)
    deployment_id = deployment['id']
    deployment_action = deployment['action']
    self.assertEqual('INIT', deployment_action)
    config_id = deployment['config_id']
    self.assertIsNotNone(config_id)
    updated = self.engine.update_software_deployment(self.ctx, deployment_id=deployment_id, config_id=config_id, input_values={}, output_values={}, action='DEPLOY', status='WAITING', status_reason='', updated_at=None)
    self.assertIsNotNone(updated)
    self.assertEqual(config_id, updated['config_id'])
    self.assertEqual('DEPLOY', updated['action'])
    self.assertEqual('WAITING', updated['status'])
    self.assertEqual(2, mock_push.call_count)