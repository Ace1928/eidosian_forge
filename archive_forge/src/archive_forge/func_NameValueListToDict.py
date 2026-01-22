import copy
import gyp.input
import argparse
import os.path
import re
import shlex
import sys
import traceback
from gyp.common import GypError
def NameValueListToDict(name_value_list):
    """
  Takes an array of strings of the form 'NAME=VALUE' and creates a dictionary
  of the pairs.  If a string is simply NAME, then the value in the dictionary
  is set to True.  If VALUE can be converted to an integer, it is.
  """
    result = {}
    for item in name_value_list:
        tokens = item.split('=', 1)
        if len(tokens) == 2:
            try:
                token_value = int(tokens[1])
            except ValueError:
                token_value = tokens[1]
            result[tokens[0]] = token_value
        else:
            result[tokens[0]] = True
    return result