import os
import sys
from troveclient.compat import common
class SecurityGroupCommands(common.AuthedCommandsBase):
    """Commands to list and show Security Groups For an Instance and
    create and delete security group rules for them.
    """
    params = ['id', 'secgroup_id', 'protocol', 'from_port', 'to_port', 'cidr']

    def get(self):
        """Get a security group associated with an instance."""
        self._require('id')
        self._pretty_print(self.dbaas.security_groups.get, self.id)

    def list(self):
        """List all the Security Groups and the rules."""
        self._pretty_paged(self.dbaas.security_groups.list)

    def add_rule(self):
        """Add a security group rule."""
        self._require('secgroup_id', 'protocol', 'from_port', 'to_port', 'cidr')
        self.dbaas.security_group_rules.create(self.secgroup_id, self.protocol, self.from_port, self.to_port, self.cidr)

    def delete_rule(self):
        """Delete a security group rule."""
        self._require('id')
        self.dbaas.security_group_rules.delete(self.id)