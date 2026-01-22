import os
import sys
from troveclient.compat import common
class DatabaseCommands(common.AuthedCommandsBase):
    """Database CRUD operations on an instance."""
    params = ['name', 'id', 'limit', 'marker']

    def create(self):
        """Create a database."""
        self._require('id', 'name')
        databases = [{'name': self.name}]
        print(self.dbaas.databases.create(self.id, databases))

    def delete(self):
        """Delete a database."""
        self._require('id', 'name')
        print(self.dbaas.databases.delete(self.id, self.name))

    def list(self):
        """List the databases."""
        self._require('id')
        self._pretty_paged(self.dbaas.databases.list, self.id)