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
def _create_projects_hierarchy(self, hierarchy_size=2, domain_id=None, is_domain=False, parent_project_id=None):
    """Create a project hierarchy with specified size.

        :param hierarchy_size: the desired hierarchy size, default is 2 -
                               a project with one child.
        :param domain_id: domain where the projects hierarchy will be created.
        :param is_domain: if the hierarchy will have the is_domain flag active
                          or not.
        :param parent_project_id: if the intention is to create a
            sub-hierarchy, sets the sub-hierarchy root. Defaults to creating
            a new hierarchy, i.e. a new root project.

        :returns projects: a list of the projects in the created hierarchy.

        """
    if domain_id is None:
        domain_id = CONF.identity.default_domain_id
    if parent_project_id:
        project = unit.new_project_ref(parent_id=parent_project_id, domain_id=domain_id, is_domain=is_domain)
    else:
        project = unit.new_project_ref(domain_id=domain_id, is_domain=is_domain)
    project_id = project['id']
    project = PROVIDERS.resource_api.create_project(project_id, project)
    projects = [project]
    for i in range(1, hierarchy_size):
        new_project = unit.new_project_ref(parent_id=project_id, domain_id=domain_id)
        PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        projects.append(new_project)
        project_id = new_project['id']
    return projects