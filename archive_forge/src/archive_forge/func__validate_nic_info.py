import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _validate_nic_info(nic_info, nic_str):
    if not bool(nic_info.get('net-id')) != bool(nic_info.get('port-id')):
        raise exceptions.ValidationError(NIC_ERROR % (_("nic='%s'") % nic_str))