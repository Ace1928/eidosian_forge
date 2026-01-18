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
def test_validate_invalid_dsversion(self):
    props = self.tmpl['resources']['cluster']['properties'].copy()
    props['datastore_version'] = '2.6.2'
    self.rsrc_defn = self.rsrc_defn.freeze(properties=props)
    tc = cluster.TroveCluster('cluster', self.rsrc_defn, self.stack)
    ex = self.assertRaises(exception.StackValidationFailed, tc.validate)
    error_msg = 'Datastore version 2.6.2 for datastore type mongodb is not valid. Allowed versions are 2.6.1.'
    self.assertEqual(error_msg, str(ex))