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
def _handle_in_progress(self, fn, *args, **kwargs):
    build_timeout = self.conf.build_timeout
    build_interval = self.conf.build_interval
    start = timeutils.utcnow()
    while timeutils.delta_seconds(start, timeutils.utcnow()) < build_timeout:
        try:
            fn(*args, **kwargs)
        except heat_exceptions.HTTPConflict as ex:
            if ex.error['error']['type'] != 'ActionInProgress':
                raise ex
            time.sleep(build_interval)
        else:
            break