from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
@classmethod
def IsDefined(cls, builtins_list, builtin_name):
    """Finds if a builtin is defined in a given list of builtin handler objects.

    Args:
      builtins_list: A list of `BuiltinHandler` objects, typically
          `yaml.builtins`.
      builtin_name: The name of the built-in that you want to determine whether
          it is defined.

    Returns:
      `True` if `builtin_name` is defined by a member of `builtins_list`; all
      other results return `False`.
    """
    for b in builtins_list:
        if b.builtin_name == builtin_name:
            return True
    return False