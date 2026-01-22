from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
class KeystoneGroupRoleAssignment(resource.Resource, KeystoneRoleAssignmentMixin):
    """Resource for granting roles to a group.

    Resource for specifying groups and their's roles.
    """
    support_status = support.SupportStatus(version='5.0.0', message=_('Supported versions: keystone v3'))
    default_client_name = 'keystone'
    PROPERTIES = GROUP, = ('group',)
    properties_schema = {GROUP: properties.Schema(properties.Schema.STRING, _('Name or id of keystone group.'), required=True, update_allowed=True, constraints=[constraints.CustomConstraint('keystone.group')])}
    properties_schema.update(KeystoneRoleAssignmentMixin.mixin_properties_schema)

    def client(self):
        return super(KeystoneGroupRoleAssignment, self).client().client

    @property
    def group_id(self):
        try:
            return self.client_plugin().get_group_id(self.properties.get(self.GROUP))
        except Exception as ex:
            self.client_plugin().ignore_not_found(ex)
            return None

    def handle_create(self):
        self.create_assignment(group_id=self.group_id)

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        self.update_assignment(group_id=self.group_id, prop_diff=prop_diff)

    def handle_delete(self):
        with self.client_plugin().ignore_not_found:
            self.delete_assignment(group_id=self.group_id)

    def validate(self):
        super(KeystoneGroupRoleAssignment, self).validate()
        self.validate_assignment_properties()