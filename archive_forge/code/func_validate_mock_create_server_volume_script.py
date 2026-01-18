from unittest import mock
from cinderclient.v3 import client as cinderclient
from heat.engine.clients.os import cinder
from heat.engine.clients.os import nova
from heat.engine.resources.aws.ec2 import volume as aws_vol
from heat.engine.resources.openstack.cinder import volume as os_vol
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def validate_mock_create_server_volume_script(self):
    self.fc.volumes.create_server_volume.assert_called_once_with(device=u'/dev/vdc', server_id=u'WikiDatabase', volume_id='vol-123')