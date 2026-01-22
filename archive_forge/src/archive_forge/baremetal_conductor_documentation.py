import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
Show baremetal conductor details