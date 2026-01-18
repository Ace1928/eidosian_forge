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
def verify_resource_status(self, stack_identifier, resource_name, status='CREATE_COMPLETE'):
    try:
        res = self.client.resources.get(stack_identifier, resource_name)
    except heat_exceptions.HTTPNotFound:
        return False
    return res.resource_status == status