import uuid
from oauthlib import oauth1
from testtools import matchers
from keystoneauth1.extras import oauth1 as ksa_oauth1
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils as test_utils
Validate data in the headers.

        Assert that the data in the headers matches the data
        that is produced from oauthlib.
        