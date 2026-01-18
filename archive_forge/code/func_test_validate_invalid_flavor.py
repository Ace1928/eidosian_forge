import copy
from unittest import mock
from troveclient import exceptions as troveexc
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine.resources.openstack.trove import cluster
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_invalid_flavor(self):
    self.troveclient.flavors.get.side_effect = troveexc.NotFound()
    self.troveclient.flavors.find.side_effect = troveexc.NotFound()
    props = copy.deepcopy(self.tmpl['resources']['cluster']['properties'])
    props['instances'][0]['flavor'] = 'm1.small'
    self.rsrc_defn = self.rsrc_defn.freeze(properties=props)
    tc = cluster.TroveCluster('cluster', self.rsrc_defn, self.stack)
    ex = self.assertRaises(exception.StackValidationFailed, tc.validate)
    error_msg = "Property error: resources.cluster.properties.instances[0].flavor: Error validating value 'm1.small': Not Found (HTTP 404)"
    self.assertEqual(error_msg, str(ex))