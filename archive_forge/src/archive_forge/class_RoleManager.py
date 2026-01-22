import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
class RoleManager(manager.Manager):
    """Default pivot point for the Role backend."""
    driver_namespace = 'keystone.role'
    _provides_api = 'role_api'
    _ROLE = 'role'

    def __init__(self):
        role_driver = CONF.role.driver
        if role_driver is None:
            assignment_driver = CONF.assignment.driver
            assignment_manager_obj = manager.load_driver(Manager.driver_namespace, assignment_driver)
            role_driver = assignment_manager_obj.default_role_driver()
        super(RoleManager, self).__init__(role_driver)

    @MEMOIZE
    def get_role(self, role_id):
        return self.driver.get_role(role_id)

    def get_unique_role_by_name(self, role_name, hints=None):
        if not hints:
            hints = driver_hints.Hints()
        hints.add_filter('name', role_name, case_sensitive=True)
        found_roles = PROVIDERS.role_api.list_roles(hints)
        if not found_roles:
            raise exception.RoleNotFound(_('Role %s is not defined') % role_name)
        elif len(found_roles) == 1:
            return {'id': found_roles[0]['id']}
        else:
            raise exception.AmbiguityError(resource='role', name=role_name)

    def create_role(self, role_id, role, initiator=None):
        role = role.copy()
        ret = self.driver.create_role(role_id, role)
        notifications.Audit.created(self._ROLE, role_id, initiator)
        if MEMOIZE.should_cache(ret):
            self.get_role.set(ret, self, role_id)
        return ret

    @manager.response_truncated
    def list_roles(self, hints=None):
        return self.driver.list_roles(hints or driver_hints.Hints())

    def _is_immutable(self, role):
        return role['options'].get(ro_opt.IMMUTABLE_OPT.option_name, False)

    def update_role(self, role_id, role, initiator=None):
        original_role = self.driver.get_role(role_id)
        ro_opt.check_immutable_update(original_resource_ref=original_role, new_resource_ref=role, type='role', resource_id=role_id)
        if 'domain_id' in role and role['domain_id'] != original_role['domain_id']:
            raise exception.ValidationError(message=_('Update of `domain_id` is not allowed.'))
        ret = self.driver.update_role(role_id, role)
        notifications.Audit.updated(self._ROLE, role_id, initiator)
        self.get_role.invalidate(self, role_id)
        return ret

    def delete_role(self, role_id, initiator=None):
        role = self.driver.get_role(role_id)
        ro_opt.check_immutable_delete(resource_ref=role, resource_type='role', resource_id=role_id)
        PROVIDERS.assignment_api._send_app_cred_notification_for_role_removal(role_id)
        PROVIDERS.assignment_api.delete_role_assignments(role_id)
        self.driver.delete_role(role_id)
        notifications.Audit.deleted(self._ROLE, role_id, initiator)
        self.get_role.invalidate(self, role_id)
        reason = 'Invalidating the token cache because role %(role_id)s has been removed. Role assignments for users will be recalculated and enforced accordingly the next time they authenticate or validate a token' % {'role_id': role_id}
        notifications.invalidate_token_cache_notification(reason)
        COMPUTED_ASSIGNMENTS_REGION.invalidate()

    def create_implied_role(self, prior_role_id, implied_role_id):
        implied_role = self.driver.get_role(implied_role_id)
        prior_role = self.driver.get_role(prior_role_id)
        if implied_role['name'] in CONF.assignment.prohibited_implied_role:
            raise exception.InvalidImpliedRole(role_id=implied_role_id)
        if prior_role['domain_id'] is None and implied_role['domain_id']:
            msg = _('Global role cannot imply a domain-specific role')
            raise exception.InvalidImpliedRole(msg, role_id=implied_role_id)
        response = self.driver.create_implied_role(prior_role_id, implied_role_id)
        COMPUTED_ASSIGNMENTS_REGION.invalidate()
        return response

    def delete_implied_role(self, prior_role_id, implied_role_id):
        self.driver.delete_implied_role(prior_role_id, implied_role_id)
        COMPUTED_ASSIGNMENTS_REGION.invalidate()