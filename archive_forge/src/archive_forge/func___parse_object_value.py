from __future__ import absolute_import
import functools
import json
import re
import sys
def __parse_object_value(self, token, string):
    self.__parse_value(token, string, Parser.__parse_object_next)