import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __NeedRequestType(self, method_description, request_type):
    """Determine if this method needs a new request type created."""
    if not request_type:
        return True
    method_id = method_description.get('id', '')
    if method_id in self.__unelidable_request_methods:
        return True
    message = self.__message_registry.LookupDescriptorOrDie(request_type)
    if message is None:
        return True
    field_names = [x.name for x in message.fields]
    parameters = method_description.get('parameters', {})
    for param_name, param_info in parameters.items():
        if param_info.get('location') != 'path' or self.__names.CleanName(param_name) not in field_names:
            break
    else:
        return False
    return True