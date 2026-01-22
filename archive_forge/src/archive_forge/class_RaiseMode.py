from threading import Event, Lock
from uuid import uuid4
from ncclient.xml_ import *
from ncclient.logging_ import SessionLoggerAdapter
from ncclient.transport import SessionListener
from ncclient.operations import util
from ncclient.operations.errors import OperationError, TimeoutExpiredError, MissingCapabilityError
import logging
class RaiseMode(object):
    """
    Define how errors indicated by RPC should be handled.

    Note that any error_filters defined in the device handler will still be
    applied, even if ERRORS or ALL is defined: If the filter matches, an exception
    will NOT be raised.

    """
    NONE = 0
    "Don't attempt to raise any type of `rpc-error` as :exc:`RPCError`."
    ERRORS = 1
    'Raise only when the `error-type` indicates it is an honest-to-god error.'
    ALL = 2
    "Don't look at the `error-type`, always raise."