import os
import sys
from troveclient.compat import common
class ConfigurationsCommands(common.AuthedCommandsBase):
    """Command to manage and show configurations."""
    params = ['name', 'instances', 'values', 'description', 'parameter']

    def get(self):
        """Get details for the specified configuration."""
        self._require('id')
        self._pretty_print(self.dbaas.configurations.get, self.id)

    def list_instances(self):
        """Get details for the specified configuration."""
        self._require('id')
        self._pretty_list(self.dbaas.configurations.instances, self.id)

    def list(self):
        """List configurations."""
        self._pretty_list(self.dbaas.configurations.list)

    def create(self):
        """Create a new configuration."""
        self._require('name', 'values')
        self._pretty_print(self.dbaas.configurations.create, self.name, self.values, self.description)

    def update(self):
        """Update an existing configuration."""
        self._require('id', 'values')
        self._pretty_print(self.dbaas.configurations.update, self.id, self.values, self.name, self.description)

    def edit(self):
        """Edit an existing configuration values."""
        self._require('id', 'values')
        self._pretty_print(self.dbaas.configurations.edit, self.id, self.values)

    def delete(self):
        """Delete a configuration."""
        self._require('id')
        self._pretty_print(self.dbaas.configurations.delete, self.id)