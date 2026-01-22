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
def _wait_for_share_status(cs, share, expected_status='available'):
    return _wait_for_resource_status(cs, share, expected_status, resource_type='share')