import uuid
from keystoneauth1 import exceptions as ks_exc
import requests.exceptions
from openstack.config import cloud_region
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
Bogus service type disabled if not in requested service_types.