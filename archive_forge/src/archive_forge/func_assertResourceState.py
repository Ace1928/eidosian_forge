from unittest import mock
from neutronclient.common import exceptions as neutron_exc
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def assertResourceState(self, rsrc, ref_id, metadata=None):
    metadata = metadata or {}
    self.assertIsNone(rsrc.validate())
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.assertEqual(ref_id, rsrc.FnGetRefId())
    self.assertEqual(metadata, dict(rsrc.metadata_get()))