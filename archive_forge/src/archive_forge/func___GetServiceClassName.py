import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __GetServiceClassName(self, service_name):
    return self.__names.ClassName('%sService' % self.__names.ClassName(service_name))