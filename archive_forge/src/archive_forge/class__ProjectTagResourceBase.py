import functools
import flask
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.resource import schema
from keystone.server import flask as ks_flask
class _ProjectTagResourceBase(ks_flask.ResourceBase):
    collection_key = 'projects'
    member_key = 'tags'
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='resource_api', method='get_project_tag')

    @classmethod
    def wrap_member(cls, ref, collection_name=None, member_name=None):
        member_name = member_name or cls.member_key
        new_ref = {'links': {'self': ks_flask.full_url()}}
        new_ref[member_name] = ref or []
        return new_ref