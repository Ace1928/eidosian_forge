from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
class InvalidRPCConnectionReuse(RPCException):
    msg_fmt = 'Invalid reuse of an RPC connection.'