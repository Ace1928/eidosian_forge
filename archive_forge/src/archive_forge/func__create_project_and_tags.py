import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.resource.backends import sql as resource_sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import utils as test_utils
def _create_project_and_tags(self, num_of_tags=1):
    """Create a project and tags associated to that project.

        :param num_of_tags: the desired number of tags attached to a
                            project, default is 1.

        :returns: A tuple of a new project and a list of random tags
        """
    tags = [uuid.uuid4().hex for i in range(num_of_tags)]
    ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, tags=tags)
    project = PROVIDERS.resource_api.create_project(ref['id'], ref)
    return (project, tags)