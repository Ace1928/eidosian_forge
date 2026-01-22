import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __CreateVoidResponseType(self, method_description):
    """Create an empty response type."""
    schema = {}
    method_name = self.__names.ClassName(method_description['id'], separator='.')
    schema['id'] = self.__names.ClassName('%sResponse' % method_name)
    schema['type'] = 'object'
    schema['description'] = 'An empty %s response.' % method_name
    self.__message_registry.AddDescriptorFromSchema(schema['id'], schema)
    return schema['id']