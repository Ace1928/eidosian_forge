import os
import threading
from oauth2client import client
def _validate_file(self):
    if os.path.islink(self._filename):
        raise CredentialsFileSymbolicLinkError('File: {0} is a symbolic link.'.format(self._filename))