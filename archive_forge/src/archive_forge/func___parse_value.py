from __future__ import absolute_import
import functools
import json
import re
import sys
def __parse_value(self, token, string, next_state):
    number_types = [int]
    number_types.extend([float])
    number_types = tuple(number_types)
    if token in [False, None, True] or isinstance(token, number_types):
        self.__put_value(token)
    elif token == 'string':
        self.__put_value(string)
    else:
        if token == '{':
            self.__push_object()
        elif token == '[':
            self.__push_array()
        else:
            self.__error('syntax error expecting value')
        return
    self.parse_state = next_state