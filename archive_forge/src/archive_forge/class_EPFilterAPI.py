import flask_restful
import http.client
from keystone.api._shared import json_home_relations
from keystone.api import endpoints as _endpoints_api
from keystone.catalog import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class EPFilterAPI(ks_flask.APIBase):
    _name = 'OS-EP-FILTER'
    _import_name = __name__
    _api_url_prefix = '/OS-EP-FILTER'
    resources = [EndpointGroupsResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=EPFilterEndpointProjectsResource, url='/endpoints/<string:endpoint_id>/projects', resource_kwargs={}, rel='endpoint_projects', resource_relation_func=_build_resource_relation, path_vars={'endpoint_id': json_home.Parameters.ENDPOINT_ID}), ks_flask.construct_resource_map(resource=EPFilterProjectsEndpointsResource, url='/projects/<string:project_id>/endpoints/<string:endpoint_id>', resource_kwargs={}, rel='project_endpoint', resource_relation_func=_build_resource_relation, path_vars={'endpoint_id': json_home.Parameters.ENDPOINT_ID, 'project_id': json_home.Parameters.PROJECT_ID}), ks_flask.construct_resource_map(resource=EPFilterProjectEndpointsListResource, url='/projects/<string:project_id>/endpoints', resource_kwargs={}, rel='project_endpoints', resource_relation_func=_build_resource_relation, path_vars={'project_id': json_home.Parameters.PROJECT_ID}), ks_flask.construct_resource_map(resource=EndpointFilterProjectEndpointGroupsListResource, url='/projects/<string:project_id>/endpoint_groups', resource_kwargs={}, rel='project_endpoint_groups', resource_relation_func=_build_resource_relation, path_vars={'project_id': json_home.Parameters.PROJECT_ID}), ks_flask.construct_resource_map(resource=EndpointFilterEPGroupsEndpoints, url='/endpoint_groups/<string:endpoint_group_id>/endpoints', resource_kwargs={}, rel='endpoints_in_endpoint_group', resource_relation_func=_build_resource_relation, path_vars={'endpoint_group_id': _ENDPOINT_GROUP_PARAMETER_RELATION}), ks_flask.construct_resource_map(resource=EndpointFilterEPGroupsProjects, url='/endpoint_groups/<string:endpoint_group_id>/projects', resource_kwargs={}, rel='projects_associated_with_endpoint_group', resource_relation_func=_build_resource_relation, path_vars={'endpoint_group_id': _ENDPOINT_GROUP_PARAMETER_RELATION}), ks_flask.construct_resource_map(resource=EPFilterGroupsProjectsResource, url='/endpoint_groups/<string:endpoint_group_id>/projects/<string:project_id>', resource_kwargs={}, rel='endpoint_group_to_project_association', resource_relation_func=_build_resource_relation, path_vars={'project_id': json_home.Parameters.PROJECT_ID, 'endpoint_group_id': _ENDPOINT_GROUP_PARAMETER_RELATION})]