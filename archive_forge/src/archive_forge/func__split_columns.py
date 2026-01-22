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
def _split_columns(columns, title=True):
    if title:
        list_of_keys = list(map(lambda x: x.strip().title(), columns.split(',')))
    else:
        list_of_keys = list(map(lambda x: x.strip().lower(), columns.split(',')))
    return list_of_keys