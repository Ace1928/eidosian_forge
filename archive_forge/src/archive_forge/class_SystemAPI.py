import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.server import flask as ks_flask
class SystemAPI(ks_flask.APIBase):
    _name = 'system'
    _import_name = __name__
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=SystemUsersListResource, url='/system/users/<string:user_id>/roles', resource_kwargs={}, rel='system_user_roles', path_vars={'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=SystemUsersResource, url='/system/users/<string:user_id>/roles/<string:role_id>', resource_kwargs={}, rel='system_user_role', path_vars={'role_id': json_home.Parameters.ROLE_ID, 'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=SystemGroupsRolesListResource, url='/system/groups/<string:group_id>/roles', resource_kwargs={}, rel='system_group_roles', path_vars={'group_id': json_home.Parameters.GROUP_ID}), ks_flask.construct_resource_map(resource=SystemGroupsRolestResource, url='/system/groups/<string:group_id>/roles/<string:role_id>', resource_kwargs={}, rel='system_group_role', path_vars={'role_id': json_home.Parameters.ROLE_ID, 'group_id': json_home.Parameters.GROUP_ID})]