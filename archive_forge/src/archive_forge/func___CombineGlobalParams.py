import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
def __CombineGlobalParams(self, global_params, default_params):
    """Combine the given params with the defaults."""
    util.Typecheck(global_params, (type(None), self.__client.params_type))
    result = self.__client.params_type()
    global_params = global_params or self.__client.params_type()
    for field in result.all_fields():
        value = global_params.get_assigned_value(field.name)
        if value is None:
            value = default_params.get_assigned_value(field.name)
        if value not in (None, [], ()):
            setattr(result, field.name, value)
    return result