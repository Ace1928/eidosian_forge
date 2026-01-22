from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
def _extract_key_value_options(args, option_name):
    result_dict = {}
    duplicate_options = []
    options = getattr(args, option_name, None)
    if options:
        for option in options:
            if '=' in option:
                key, value = option.split('=', 1)
            else:
                key = option
                value = None
            if key not in result_dict:
                result_dict[key] = value
            else:
                duplicate_options.append(key)
        if len(duplicate_options) > 0:
            duplicate_str = ', '.join(duplicate_options)
            msg = 'Following options were duplicated: %s' % duplicate_str
            raise exceptions.CommandError(msg)
    return result_dict