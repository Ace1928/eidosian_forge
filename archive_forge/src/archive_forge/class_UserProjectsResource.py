import base64
import secrets
import uuid
import flask
import http.client
from oslo_serialization import jsonutils
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.application_credential import schema as app_cred_schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
import keystone.conf
from keystone import exception as ks_exception
from keystone.i18n import _
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
class UserProjectsResource(ks_flask.ResourceBase):
    collection_key = 'projects'
    member_key = 'project'
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='resource_api', method='get_project')

    def get(self, user_id):
        filters = ('domain_id', 'enabled', 'name')
        ENFORCER.enforce_call(action='identity:list_user_projects', filters=filters, build_target=_build_user_target_enforcement)
        hints = self.build_driver_hints(filters)
        refs = PROVIDERS.assignment_api.list_projects_for_user(user_id)
        return self.wrap_collection(refs, hints=hints)