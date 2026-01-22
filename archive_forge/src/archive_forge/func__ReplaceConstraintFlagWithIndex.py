from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _ReplaceConstraintFlagWithIndex(arguments):
    positional_index = 0
    for i, arg in enumerate(arguments):
        if isinstance(arg, int):
            pass
        elif arg.is_group:
            _ReplaceConstraintFlagWithIndex(arg.arguments)
        elif arg.is_positional:
            positional_index -= 1
            arguments[i] = positional_index
        else:
            try:
                arguments[i] = all_flags[_FlagIndexKey(arg)].index
            except KeyError:
                pass