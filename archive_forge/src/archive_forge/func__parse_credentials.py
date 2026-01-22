import os
import re
import warnings
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.remotefs import remotefs
def _parse_credentials(self, opts_str):
    if not opts_str:
        return (None, None)
    match = self._username_regex.findall(opts_str)
    username = match[0] if match and match[0] != 'guest' else None
    match = self._password_regex.findall(opts_str)
    password = match[0] if match else None
    return (username, password)