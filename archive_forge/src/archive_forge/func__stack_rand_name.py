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
def _stack_rand_name(self):
    test_name = self.id()
    if test_name and '.' in test_name:
        name = '-'.join(test_name.split('.')[-2:])
        name = name.split('(')[0]
    else:
        name = self.__name__
    return rand_name(name)