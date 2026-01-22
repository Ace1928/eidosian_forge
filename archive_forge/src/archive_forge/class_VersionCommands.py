import os
import sys
from troveclient.compat import common
class VersionCommands(common.AuthedCommandsBase):
    """List available versions."""
    params = ['url']

    def list(self):
        """List all the supported versions."""
        self._require('url')
        self._pretty_list(self.dbaas.versions.index, self.url)