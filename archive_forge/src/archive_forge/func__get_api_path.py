from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def _get_api_path(self, path, uuid=None):
    """
        This function returns the full url from relative path and uuid.
        """
    if path == 'logout':
        return self.prefix + '/' + path
    elif uuid:
        return self.prefix + '/api/' + path + '/' + uuid
    else:
        return self.prefix + '/api/' + path