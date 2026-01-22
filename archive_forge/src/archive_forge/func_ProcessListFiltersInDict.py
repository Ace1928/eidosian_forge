import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def ProcessListFiltersInDict(name, the_dict):
    """Process regular expression and exclusion-based filters on lists.

  An exclusion list is in a dict key named with a trailing "!", like
  "sources!".  Every item in such a list is removed from the associated
  main list, which in this example, would be "sources".  Removed items are
  placed into a "sources_excluded" list in the dict.

  Regular expression (regex) filters are contained in dict keys named with a
  trailing "/", such as "sources/" to operate on the "sources" list.  Regex
  filters in a dict take the form:
    'sources/': [ ['exclude', '_(linux|mac|win)\\.cc$'],
                  ['include', '_mac\\.cc$'] ],
  The first filter says to exclude all files ending in _linux.cc, _mac.cc, and
  _win.cc.  The second filter then includes all files ending in _mac.cc that
  are now or were once in the "sources" list.  Items matching an "exclude"
  filter are subject to the same processing as would occur if they were listed
  by name in an exclusion list (ending in "!").  Items matching an "include"
  filter are brought back into the main list if previously excluded by an
  exclusion list or exclusion regex filter.  Subsequent matching "exclude"
  patterns can still cause items to be excluded after matching an "include".
  """
    lists = []
    del_lists = []
    for key, value in the_dict.items():
        operation = key[-1]
        if operation != '!' and operation != '/':
            continue
        if type(value) is not list:
            raise ValueError(name + ' key ' + key + ' must be list, not ' + value.__class__.__name__)
        list_key = key[:-1]
        if list_key not in the_dict:
            del_lists.append(key)
            continue
        if type(the_dict[list_key]) is not list:
            value = the_dict[list_key]
            raise ValueError(name + ' key ' + list_key + ' must be list, not ' + value.__class__.__name__ + ' when applying ' + {'!': 'exclusion', '/': 'regex'}[operation])
        if list_key not in lists:
            lists.append(list_key)
    for del_list in del_lists:
        del the_dict[del_list]
    for list_key in lists:
        the_list = the_dict[list_key]
        list_actions = list((-1,) * len(the_list))
        exclude_key = list_key + '!'
        if exclude_key in the_dict:
            for exclude_item in the_dict[exclude_key]:
                for index, list_item in enumerate(the_list):
                    if exclude_item == list_item:
                        list_actions[index] = 0
            del the_dict[exclude_key]
        regex_key = list_key + '/'
        if regex_key in the_dict:
            for regex_item in the_dict[regex_key]:
                [action, pattern] = regex_item
                pattern_re = re.compile(pattern)
                if action == 'exclude':
                    action_value = 0
                elif action == 'include':
                    action_value = 1
                else:
                    raise ValueError('Unrecognized action ' + action + ' in ' + name + ' key ' + regex_key)
                for index, list_item in enumerate(the_list):
                    if list_actions[index] == action_value:
                        continue
                    if pattern_re.search(list_item):
                        list_actions[index] = action_value
            del the_dict[regex_key]
        excluded_key = list_key + '_excluded'
        if excluded_key in the_dict:
            raise GypError(name + ' key ' + excluded_key + ' must not be present prior  to applying exclusion/regex filters for ' + list_key)
        excluded_list = []
        for index in range(len(list_actions) - 1, -1, -1):
            if list_actions[index] == 0:
                excluded_list.insert(0, the_list[index])
                del the_list[index]
        if len(excluded_list) > 0:
            the_dict[excluded_key] = excluded_list
    for key, value in the_dict.items():
        if type(value) is dict:
            ProcessListFiltersInDict(key, value)
        elif type(value) is list:
            ProcessListFiltersInList(key, value)