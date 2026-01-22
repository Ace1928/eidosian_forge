import os
import sys
from troveclient.compat import common
class RootCommands(common.AuthedCommandsBase):
    """Root user related operations on an instance."""
    params = ['id']

    def create(self):
        """Enable the instance's root user."""
        self._require('id')
        try:
            user, password = self.dbaas.root.create(self.id)
            print('User:\t\t%s\nPassword:\t%s' % (user, password))
        except Exception:
            print(sys.exc_info()[1])

    def delete(self):
        """Disable the instance's root user."""
        self._require('id')
        print(self.dbaas.root.delete(self.id))

    def enabled(self):
        """Check the instance for root access."""
        self._require('id')
        self._pretty_print(self.dbaas.root.is_root_enabled, self.id)