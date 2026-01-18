import contextlib
import os
import re
import textwrap
import time
from urllib import parse
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import prettytable
from novaclient import exceptions
from novaclient.i18n import _
def validate_flavor_metadata_keys(keys):
    for key in keys:
        valid_name = VALID_KEY_REGEX.match(key)
        if not valid_name:
            msg = _('Invalid key: "%s". Keys may only contain letters, numbers, spaces, underscores, periods, colons and hyphens.')
            raise exceptions.CommandError(msg % key)