from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def _get_db_instance(self):
    t = template_format.parse(db_template)
    res = self._setup_test_instance('trove_check', t)
    res.state_set(res.CREATE, res.COMPLETE)
    res.flavor = 'Foo Flavor'
    res.volume = 'Foo Volume'
    res.datastore_type = 'Foo Type'
    res.datastore_version = 'Foo Version'
    return res