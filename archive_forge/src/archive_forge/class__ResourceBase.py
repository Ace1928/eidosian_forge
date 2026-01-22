import flask
import flask_restful
import http.client
from oslo_serialization import jsonutils
from oslo_log import log
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import schema
from keystone.federation import utils
from keystone.server import flask as ks_flask
class _ResourceBase(ks_flask.ResourceBase):
    json_home_resource_rel_func = _build_resource_relation
    json_home_parameter_rel_func = _build_param_relation

    @classmethod
    def wrap_member(cls, ref, collection_name=None, member_name=None):
        cls._add_self_referential_link(ref, collection_name)
        cls._add_related_links(ref)
        return {member_name or cls.member_key: cls.filter_params(ref)}

    @staticmethod
    def _add_related_links(ref):
        pass