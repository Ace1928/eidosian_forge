import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _set_default_assignment_attributes(self, **attribs):
    """Insert default values for missing attributes of role assignment.

        If no actor, target or role are provided, they will default to values
        from sample data.

        :param attribs: info from a role assignment entity. Valid attributes
                        are: role_id, domain_id, project_id, group_id, user_id
                        and inherited_to_projects.

        """
    if not any((target in attribs for target in ('domain_id', 'projects_id'))):
        attribs['project_id'] = self.project_id
    if not any((actor in attribs for actor in ('user_id', 'group_id'))):
        attribs['user_id'] = self.default_user_id
    if 'role_id' not in attribs:
        attribs['role_id'] = self.role_id
    return attribs