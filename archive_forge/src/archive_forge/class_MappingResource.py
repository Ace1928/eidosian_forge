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
class MappingResource(_ResourceBase):
    collection_key = 'mappings'
    member_key = 'mapping'
    api_prefix = '/OS-FEDERATION'

    def get(self, mapping_id=None):
        if mapping_id is not None:
            return self._get_mapping(mapping_id)
        return self._list_mappings()

    def _get_mapping(self, mapping_id):
        """Get a mapping.

        HEAD/GET /OS-FEDERATION/mappings/{mapping_id}
        """
        ENFORCER.enforce_call(action='identity:get_mapping')
        return self.wrap_member(PROVIDERS.federation_api.get_mapping(mapping_id))

    def _list_mappings(self):
        """List mappings.

        HEAD/GET /OS-FEDERATION/mappings
        """
        ENFORCER.enforce_call(action='identity:list_mappings')
        return self.wrap_collection(PROVIDERS.federation_api.list_mappings())

    def _internal_normalize_and_validate_attribute_mapping(self, action_executed_message='created'):
        mapping = self.request_body_json.get('mapping', {})
        mapping = self._normalize_dict(mapping)
        if not mapping.get('schema_version'):
            default_schema_version = utils.get_default_attribute_mapping_schema_version()
            LOG.debug("A mapping [%s] was %s without providing a 'schema_version'; therefore, we need to set one. The current default is [%s]. We will use this value for the attribute mapping being registered. It is recommended that one does not rely on this default value, as it can change, and the already persisted attribute mappings will remain with the previous default values.", mapping, action_executed_message, default_schema_version)
            mapping['schema_version'] = default_schema_version
        utils.validate_mapping_structure(mapping)
        return mapping

    def put(self, mapping_id):
        """Create a mapping.

        PUT /OS-FEDERATION/mappings/{mapping_id}
        """
        ENFORCER.enforce_call(action='identity:create_mapping')
        am = self._internal_normalize_and_validate_attribute_mapping('registered')
        mapping_ref = PROVIDERS.federation_api.create_mapping(mapping_id, am)
        return (self.wrap_member(mapping_ref), http.client.CREATED)

    def patch(self, mapping_id):
        """Update an attribute mapping for identity federation.

        PATCH /OS-FEDERATION/mappings/{mapping_id}
        """
        ENFORCER.enforce_call(action='identity:update_mapping')
        am = self._internal_normalize_and_validate_attribute_mapping('updated')
        mapping_ref = PROVIDERS.federation_api.update_mapping(mapping_id, am)
        return self.wrap_member(mapping_ref)

    def delete(self, mapping_id):
        """Delete a mapping.

        DELETE /OS-FEDERATION/mappings/{mapping_id}
        """
        ENFORCER.enforce_call(action='identity:delete_mapping')
        PROVIDERS.federation_api.delete_mapping(mapping_id)
        return (None, http.client.NO_CONTENT)