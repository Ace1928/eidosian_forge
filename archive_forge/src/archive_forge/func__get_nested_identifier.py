import os
import random
import re
import subprocess
import time
import urllib
import fixtures
from heatclient import exc as heat_exceptions
from keystoneauth1 import exceptions as kc_exceptions
from oslo_log import log as logging
from oslo_utils import timeutils
from tempest import config
import testscenarios
import testtools
from heat_integrationtests.common import clients
from heat_integrationtests.common import exceptions
def _get_nested_identifier(self, stack_identifier, res_name):
    rsrc = self.client.resources.get(stack_identifier, res_name)
    nested_link = [lk for lk in rsrc.links if lk['rel'] == 'nested']
    nested_href = nested_link[0]['href']
    nested_id = nested_href.split('/')[-1]
    nested_identifier = '/'.join(nested_href.split('/')[-2:])
    self.assertEqual(rsrc.physical_resource_id, nested_id)
    nested_stack = self.client.stacks.get(nested_id, resolve_outputs=False)
    nested_identifier2 = '%s/%s' % (nested_stack.stack_name, nested_stack.id)
    self.assertEqual(nested_identifier, nested_identifier2)
    parent_id = stack_identifier.split('/')[-1]
    self.assertEqual(parent_id, nested_stack.parent)
    return nested_identifier