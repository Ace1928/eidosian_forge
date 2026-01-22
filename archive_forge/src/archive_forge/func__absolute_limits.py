import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def _absolute_limits(self):
    max_personality = mock.Mock()
    max_personality.name = 'maxPersonality'
    max_personality.value = 5
    max_personality_size = mock.Mock()
    max_personality_size.name = 'maxPersonalitySize'
    max_personality_size.value = 10240
    max_server_meta = mock.Mock()
    max_server_meta.name = 'maxServerMeta'
    max_server_meta.value = 3
    yield max_personality
    yield max_personality_size
    yield max_server_meta