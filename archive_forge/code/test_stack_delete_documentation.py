import copy
import time
from unittest import mock
import fixtures
from keystoneauth1 import exceptions as kc_exceptions
from oslo_log import log as logging
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os.keystone import heat_keystoneclient as hkc
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.objects import user_creds as ucreds_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
Do not stop deleting stacks even failed deleting user_creds.

        It may happen that user_creds were incorrectly saved (truncated) and
        thus cannot be correctly retrieved (and decrypted). In this case,
        stack delete should not be stopped.
        