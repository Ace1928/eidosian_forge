import copy
import fixtures
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from testtools import matchers
from keystoneclient import access
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
from keystoneclient.v3.contrib.federation import base
from keystoneclient.v3.contrib.federation import identity_providers
from keystoneclient.v3.contrib.federation import mappings
from keystoneclient.v3.contrib.federation import protocols
from keystoneclient.v3.contrib.federation import service_providers
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
def test_positional_parameters_expect_fail(self):
    """Ensure CrudManager raises TypeError exceptions.

        After passing wrong number of positional arguments
        an exception should be raised.

        Operations to be tested:
            * create()
            * get()
            * list()
            * delete()
            * update()

        """
    POS_PARAM_1 = uuid.uuid4().hex
    POS_PARAM_2 = uuid.uuid4().hex
    POS_PARAM_3 = uuid.uuid4().hex
    PARAMETERS = {'create': (POS_PARAM_1, POS_PARAM_2), 'get': (POS_PARAM_1, POS_PARAM_2), 'list': (POS_PARAM_1, POS_PARAM_2), 'update': (POS_PARAM_1, POS_PARAM_2, POS_PARAM_3), 'delete': (POS_PARAM_1, POS_PARAM_2)}
    for f_name, args in PARAMETERS.items():
        self.assertRaises(TypeError, getattr(self.manager, f_name), *args)