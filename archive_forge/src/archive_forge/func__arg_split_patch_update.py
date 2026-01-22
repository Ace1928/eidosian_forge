import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def _arg_split_patch_update(arg, patch=False):
    if patch:
        arg = ','.join(arg)
    if not arg or arg == '[]':
        arg_split = []
    else:
        arg_split = arg.split(',')
    return arg_split