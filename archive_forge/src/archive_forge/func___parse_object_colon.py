from __future__ import absolute_import
import functools
import json
import re
import sys
def __parse_object_colon(self, token, unused_string):
    if token == ':':
        self.parse_state = Parser.__parse_object_value
    else:
        self.__error("syntax error parsing object expecting ':'")