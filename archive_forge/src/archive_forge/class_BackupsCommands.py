import os
import sys
from troveclient.compat import common
class BackupsCommands(common.AuthedCommandsBase):
    """Command to manage and show backups."""
    params = ['name', 'instance', 'description']

    def get(self):
        """Get details for the specified backup."""
        self._require('id')
        self._pretty_print(self.dbaas.backups.get, self.id)

    def list(self):
        """List backups."""
        self._pretty_list(self.dbaas.backups.list)

    def create(self):
        """Create a new backup."""
        self._require('name', 'instance')
        self._pretty_print(self.dbaas.backups.create, self.name, self.instance, self.description)

    def delete(self):
        """Delete a backup."""
        self._require('id')
        self._pretty_print(self.dbaas.backups.delete, self.id)