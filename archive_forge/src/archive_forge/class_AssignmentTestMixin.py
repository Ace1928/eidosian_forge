import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
class AssignmentTestMixin(object):
    """To hold assignment helper functions."""

    def build_role_assignment_query_url(self, effective=False, **filters):
        """Build and return a role assignment query url with provided params.

        Available filters are: domain_id, project_id, user_id, group_id,
        role_id and inherited_to_projects.
        """
        query_params = '?effective' if effective else ''
        for k, v in filters.items():
            query_params += '?' if not query_params else '&'
            if k == 'inherited_to_projects':
                query_params += 'scope.OS-INHERIT:inherited_to=projects'
            else:
                if k in ['domain_id', 'project_id']:
                    query_params += 'scope.'
                elif k not in ['user_id', 'group_id', 'role_id']:
                    raise ValueError("Invalid key '%s' in provided filters." % k)
                query_params += '%s=%s' % (k.replace('_', '.'), v)
        return '/role_assignments%s' % query_params

    def build_role_assignment_link(self, **attribs):
        """Build and return a role assignment link with provided attributes.

        Provided attributes are expected to contain: domain_id or project_id,
        user_id or group_id, role_id and, optionally, inherited_to_projects.
        """
        if attribs.get('domain_id'):
            link = '/domains/' + attribs['domain_id']
        elif attribs.get('system'):
            link = '/system'
        else:
            link = '/projects/' + attribs['project_id']
        if attribs.get('user_id'):
            link += '/users/' + attribs['user_id']
        else:
            link += '/groups/' + attribs['group_id']
        link += '/roles/' + attribs['role_id']
        if attribs.get('inherited_to_projects'):
            return '/OS-INHERIT%s/inherited_to_projects' % link
        return link

    def build_role_assignment_entity(self, link=None, prior_role_link=None, **attribs):
        """Build and return a role assignment entity with provided attributes.

        Provided attributes are expected to contain: domain_id or project_id,
        user_id or group_id, role_id and, optionally, inherited_to_projects.
        """
        entity = {'links': {'assignment': link or self.build_role_assignment_link(**attribs)}}
        if attribs.get('domain_id'):
            entity['scope'] = {'domain': {'id': attribs['domain_id']}}
        elif attribs.get('system'):
            entity['scope'] = {'system': {'all': True}}
        else:
            entity['scope'] = {'project': {'id': attribs['project_id']}}
        if attribs.get('user_id'):
            entity['user'] = {'id': attribs['user_id']}
            if attribs.get('group_id'):
                entity['links']['membership'] = '/groups/%s/users/%s' % (attribs['group_id'], attribs['user_id'])
        else:
            entity['group'] = {'id': attribs['group_id']}
        entity['role'] = {'id': attribs['role_id']}
        if attribs.get('inherited_to_projects'):
            entity['scope']['OS-INHERIT:inherited_to'] = 'projects'
        if prior_role_link:
            entity['links']['prior_role'] = prior_role_link
        return entity

    def build_role_assignment_entity_include_names(self, domain_ref=None, role_ref=None, group_ref=None, user_ref=None, project_ref=None, inherited_assignment=None):
        """Build and return a role assignment entity with provided attributes.

        The expected attributes are: domain_ref or project_ref,
        user_ref or group_ref, role_ref and, optionally, inherited_to_projects.
        """
        entity = {'links': {}}
        attributes_for_links = {}
        if project_ref:
            dmn_name = PROVIDERS.resource_api.get_domain(project_ref['domain_id'])['name']
            entity['scope'] = {'project': {'id': project_ref['id'], 'name': project_ref['name'], 'domain': {'id': project_ref['domain_id'], 'name': dmn_name}}}
            attributes_for_links['project_id'] = project_ref['id']
        else:
            entity['scope'] = {'domain': {'id': domain_ref['id'], 'name': domain_ref['name']}}
            attributes_for_links['domain_id'] = domain_ref['id']
        if user_ref:
            dmn_name = PROVIDERS.resource_api.get_domain(user_ref['domain_id'])['name']
            entity['user'] = {'id': user_ref['id'], 'name': user_ref['name'], 'domain': {'id': user_ref['domain_id'], 'name': dmn_name}}
            attributes_for_links['user_id'] = user_ref['id']
        else:
            dmn_name = PROVIDERS.resource_api.get_domain(group_ref['domain_id'])['name']
            entity['group'] = {'id': group_ref['id'], 'name': group_ref['name'], 'domain': {'id': group_ref['domain_id'], 'name': dmn_name}}
            attributes_for_links['group_id'] = group_ref['id']
        if role_ref:
            entity['role'] = {'id': role_ref['id'], 'name': role_ref['name']}
            if role_ref['domain_id']:
                dmn_name = PROVIDERS.resource_api.get_domain(role_ref['domain_id'])['name']
                entity['role']['domain'] = {'id': role_ref['domain_id'], 'name': dmn_name}
            attributes_for_links['role_id'] = role_ref['id']
        if inherited_assignment:
            entity['scope']['OS-INHERIT:inherited_to'] = 'projects'
            attributes_for_links['inherited_to_projects'] = True
        entity['links']['assignment'] = self.build_role_assignment_link(**attributes_for_links)
        return entity