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
def _initiator_get_text(self, *arg, **kwargs):
    text = '## DO NOT EDIT OR REMOVE THIS FILE!\n## If you remove this file, the iSCSI daemon will not start.\n## If you change the InitiatorName, existing access control lists\n## may reject this initiator.  The InitiatorName must be unique\n## for each iSCSI initiator.  Do NOT duplicate iSCSI InitiatorNames.\nInitiatorName=%s' % self._fake_iqn
    return (text, None)