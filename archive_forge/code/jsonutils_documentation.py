import codecs
import datetime
import functools
import inspect
import io
import itertools
import json
import uuid
from xmlrpc import client as xmlrpclib
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import timeutils
Deserialize ``fp`` to a Python object.

    :param fp: a ``.read()`` -supporting file-like object
    :param encoding: encoding used to interpret the string
    :param kwargs: extra named parameters, please see documentation     of `json.loads <https://docs.python.org/2/library/json.html#basic-usage>`_
    :returns: python object
    