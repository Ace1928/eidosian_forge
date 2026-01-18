from requests_mock.contrib import fixture as rm_fixture
from glanceclient import client
from glanceclient.tests.unit.v2.fixtures import image_create_fixture
from glanceclient.tests.unit.v2.fixtures import image_list_fixture
from glanceclient.tests.unit.v2.fixtures import image_show_fixture
from glanceclient.tests.unit.v2.fixtures import schema_fixture
from glanceclient.tests import utils as testutils
from glanceclient.v2.image_schema import _BASE_SCHEMA
Client tests using the requests mock library.