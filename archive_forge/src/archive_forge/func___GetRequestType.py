import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __GetRequestType(self, body_type):
    return self.__names.ClassName(body_type.get('$ref'))