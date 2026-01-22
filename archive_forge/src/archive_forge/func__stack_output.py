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
@staticmethod
def _stack_output(stack, output_key, validate_errors=True):
    """Return a stack output value for a given key."""
    value = None
    for o in stack.outputs:
        if validate_errors and 'output_error' in o:
            raise ValueError('Unexpected output errors in %s : %s' % (output_key, o['output_error']))
        if o['output_key'] == output_key:
            value = o['output_value']
    return value