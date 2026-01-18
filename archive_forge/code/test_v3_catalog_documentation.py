import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
Deleting a project should not result in an 500 ISE.

        Deleting a project will create a notification, which the EndpointFilter
        functionality will use to clean up any project->endpoint and
        project->endpoint_group relationships. The templated catalog does not
        support such relationships, but the act of attempting to delete them
        should not cause a NotImplemented exception to be exposed to an API
        caller.

        Deleting an endpoint has a similar notification and clean up
        mechanism, but since we do not allow deletion of endpoints with the
        templated catalog, there is no testing to do for that action.
        