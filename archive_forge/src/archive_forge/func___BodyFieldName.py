import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __BodyFieldName(self, body_type):
    if body_type is None:
        return ''
    return self.__names.FieldName(body_type['$ref'])