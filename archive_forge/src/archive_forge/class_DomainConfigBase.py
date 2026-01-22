import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.resource import schema
from keystone.server import flask as ks_flask
class DomainConfigBase(ks_flask.ResourceBase):
    member_key = 'config'

    def get(self, domain_id=None, group=None, option=None):
        """Check if config option exists.

        GET/HEAD /v3/domains/{domain_id}/config
        GET/HEAD /v3/domains/{domain_id}/config/{group}
        GET/HEAD /v3/domains/{domain_id}/config/{group}/{option}
        """
        err = None
        config = {}
        try:
            PROVIDERS.resource_api.get_domain(domain_id)
        except Exception as e:
            err = e
        finally:
            if group and group == 'security_compliance':
                config = self._get_security_compliance_config(domain_id, group, option)
            else:
                config = self._get_config(domain_id, group, option)
        if err is not None:
            raise err
        return {self.member_key: config}

    def _get_config(self, domain_id, group, option):
        ENFORCER.enforce_call(action='identity:get_domain_config')
        return PROVIDERS.domain_config_api.get_config(domain_id, group=group, option=option)

    def _get_security_compliance_config(self, domain_id, group, option):
        ENFORCER.enforce_call(action='identity:get_security_compliance_domain_config')
        return PROVIDERS.domain_config_api.get_security_compliance_config(domain_id, group, option=option)

    def patch(self, domain_id=None, group=None, option=None):
        """Update domain config option.

        PATCH /v3/domains/{domain_id}/config
        PATCH /v3/domains/{domain_id}/config/{group}
        PATCH /v3/domains/{domain_id}/config/{group}/{option}
        """
        ENFORCER.enforce_call(action='identity:update_domain_config')
        PROVIDERS.resource_api.get_domain(domain_id)
        config = self.request_body_json.get('config', {})
        ref = PROVIDERS.domain_config_api.update_config(domain_id, config, group, option=option)
        return {self.member_key: ref}

    def delete(self, domain_id=None, group=None, option=None):
        """Delete domain config.

        DELETE /v3/domains/{domain_id}/config
        DELETE /v3/domains/{domain_id}/config/{group}
        DELETE /v3/domains/{domain_id}/config/{group}/{option}
        """
        ENFORCER.enforce_call(action='identity:delete_domain_config')
        PROVIDERS.resource_api.get_domain(domain_id)
        PROVIDERS.domain_config_api.delete_config(domain_id, group, option=option)
        return (None, http.client.NO_CONTENT)