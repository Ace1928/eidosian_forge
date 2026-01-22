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
def __ConstructQueryParams(self, query_params, request, global_params):
    """Construct a dictionary of query parameters for this request."""
    global_params = self.__CombineGlobalParams(global_params, self.__client.global_params)
    global_param_names = util.MapParamNames([x.name for x in self.__client.params_type.all_fields()], self.__client.params_type)
    global_params_type = type(global_params)
    query_info = dict(((param, self.__FinalUrlValue(getattr(global_params, param), getattr(global_params_type, param))) for param in global_param_names))
    query_param_names = util.MapParamNames(query_params, type(request))
    request_type = type(request)
    query_info.update(((param, self.__FinalUrlValue(getattr(request, param, None), getattr(request_type, param))) for param in query_param_names))
    query_info = dict(((k, v) for k, v in query_info.items() if v is not None))
    query_info = self.__EncodePrettyPrint(query_info)
    query_info = util.MapRequestParams(query_info, type(request))
    return query_info