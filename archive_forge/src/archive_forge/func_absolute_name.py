from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def absolute_name(self, name):
    """
        Returns true if the name includes a scheme
        """
    if name is None:
        return False
    return self._absolute_re.search(name)