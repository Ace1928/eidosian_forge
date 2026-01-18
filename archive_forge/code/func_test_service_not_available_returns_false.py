import collections
import datetime
import itertools
import json
import os
import sys
from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import attributes
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import clients
from heat.engine import constraints
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import node_data
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources.openstack.heat import none_resource
from heat.engine.resources.openstack.heat import test_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.engine import translation
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_object
from heat.objects import resource_properties_data as rpd_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
import neutronclient.common.exceptions as neutron_exp
def test_service_not_available_returns_false(self):
    """Test when the service is not in service catalog.

        When the service is not deployed, make sure resource is throwing
        ResourceTypeUnavailable exception.
        """
    with mock.patch.object(generic_rsrc.ResourceWithDefaultClientName, 'is_service_available') as mock_method:
        mock_method.return_value = (False, 'Service endpoint not in service catalog.')
        definition = rsrc_defn.ResourceDefinition(name='Test Resource', resource_type='UnavailableResourceType')
        mock_stack = mock.MagicMock()
        mock_stack.in_convergence_check = False
        mock_stack.db_resource_get.return_value = None
        rsrc = generic_rsrc.ResourceWithDefaultClientName('test_res', definition, mock_stack)
        ex = self.assertRaises(exception.ResourceTypeUnavailable, rsrc.validate_template)
        msg = 'HEAT-E99001 Service sample is not available for resource type UnavailableResourceType, reason: Service endpoint not in service catalog.'
        self.assertEqual(msg, str(ex), 'invalid exception message')
        mock_method.assert_called_once_with(mock_stack.context)