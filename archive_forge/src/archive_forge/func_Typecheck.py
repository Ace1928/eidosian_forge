import os
import random
import six
from six.moves import http_client
import six.moves.urllib.error as urllib_error
import six.moves.urllib.parse as urllib_parse
import six.moves.urllib.request as urllib_request
from apitools.base.protorpclite import messages
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
def Typecheck(arg, arg_type, msg=None):
    if not isinstance(arg, arg_type):
        if msg is None:
            if isinstance(arg_type, tuple):
                msg = 'Type of arg is "%s", not one of %r' % (type(arg), arg_type)
            else:
                msg = 'Type of arg is "%s", not "%s"' % (type(arg), arg_type)
        raise exceptions.TypecheckError(msg)
    return arg