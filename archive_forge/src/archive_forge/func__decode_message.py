import asyncore
import binascii
import collections
import errno
import functools
import hashlib
import hmac
import math
import os
import pickle
import socket
import struct
import time
import futurist
from oslo_utils import excutils
from taskflow.engines.action_engine import executor as base
from taskflow import logging
from taskflow import task as ta
from taskflow.types import notifier as nt
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.utils import schema_utils as su
from taskflow.utils import threading_utils
def _decode_message(auth_key, message, message_mac):
    tmp_message_mac = _calculate_hmac(auth_key, message)
    if tmp_message_mac != message_mac:
        raise BadHmacValueError('Invalid message hmac')
    return pickle.loads(message)