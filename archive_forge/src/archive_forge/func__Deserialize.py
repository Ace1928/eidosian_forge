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
def _Deserialize(tree):
    """Returns the deserialization of a serialized CLI tree."""
    all_flags_list = tree.get(_LOOKUP_SERIALIZED_FLAG_LIST)
    if not all_flags_list:
        return tree
    tree[_LOOKUP_SERIALIZED_FLAG_LIST] = None
    del tree[_LOOKUP_SERIALIZED_FLAG_LIST]

    def _ReplaceConstraintIndexWithArgReference(arguments, positionals):
        for i, arg in enumerate(arguments):
            if isinstance(arg, int):
                if arg < 0:
                    arguments[i] = positionals[-(arg + 1)]
                else:
                    arguments[i] = all_flags_list[arg]
            elif arg.get(LOOKUP_IS_GROUP, False):
                _ReplaceConstraintIndexWithArgReference(arg.get(LOOKUP_ARGUMENTS), positionals)

    def _ReplaceIndexWithFlagReference(command):
        flags = command[LOOKUP_FLAGS]
        for name, index in six.iteritems(flags):
            flags[name] = all_flags_list[index]
        arguments = command[LOOKUP_CONSTRAINTS][LOOKUP_ARGUMENTS]
        _ReplaceConstraintIndexWithArgReference(arguments, command[LOOKUP_POSITIONALS])
        for subcommand in command[LOOKUP_COMMANDS].values():
            _ReplaceIndexWithFlagReference(subcommand)
    _ReplaceIndexWithFlagReference(tree)
    return tree