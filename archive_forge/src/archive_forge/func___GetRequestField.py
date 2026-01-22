import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __GetRequestField(self, method_description, body_type):
    """Determine the request field for this method."""
    body_field_name = self.__BodyFieldName(body_type)
    if body_field_name in method_description.get('parameters', {}):
        body_field_name = self.__names.FieldName('%s_resource' % body_field_name)
    while body_field_name in method_description.get('parameters', {}):
        body_field_name = self.__names.FieldName('%s_body' % body_field_name)
    return body_field_name