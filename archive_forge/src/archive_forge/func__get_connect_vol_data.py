import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
@staticmethod
def _get_connect_vol_data():
    return {'stop_connecting': False, 'num_logins': 0, 'failed_logins': 0, 'stopped_threads': 0, 'found_devices': [], 'just_added_devices': []}