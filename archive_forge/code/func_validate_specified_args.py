from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import argparse
import collections
import io
import itertools
import os
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base  # pylint: disable=unused-import
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import suggest_commands
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
import six
def validate_specified_args(self, ai, specified_args, namespace, is_required=True, top=True):
    """Validate specified args against the arg group constraints.

    Each group may be mutually exclusive and/or required. Each argument may be
    required.

    Args:
      ai: ArgumentInterceptor, The argument interceptor containing the
        ai.arguments argument group.
      specified_args: set, The dests of the specified args.
      namespace: object, The parsed args namespace.
      is_required: bool, True if all containing groups are required.
      top: bool, True if ai.arguments is the top level group.

    Raises:
      ModalGroupError: If modal arg not specified.
      OptionalMutexError: On optional mutex group conflict.
      RequiredError: If required arg not specified.
      RequiredMutexError: On required mutex group conflict.

    Returns:
      True if the subgroup was specified.
    """
    also_optional = []
    have_optional = []
    have_required = []
    need_required = []
    arguments = sorted(ai.arguments, key=usage_text.GetArgSortKey) if ai.sort_args else ai.arguments
    for arg in arguments:
        if arg.is_group:
            arg_was_specified = self.validate_specified_args(arg, specified_args, namespace, is_required=is_required and arg.is_required, top=False)
        else:
            arg_was_specified = arg.dest in specified_args
        if arg_was_specified:
            if arg.is_required:
                have_required.append(arg)
            else:
                have_optional.append(arg)
        elif arg.is_required:
            if not isinstance(arg, DynamicPositionalAction):
                need_required.append(arg)
        else:
            also_optional.append(arg)
    if need_required:
        if top or (have_required and (not (have_optional or also_optional))):
            need_args = usage_text.ArgumentWrapper(arguments=need_required, is_group=True, sort_args=ai.sort_args)
            self._Error(parser_errors.RequiredError(parser=self, argument=usage_text.GetArgUsage(need_args, value=False, hidden=True, top=top)))
        if have_optional or have_required:
            have_args = usage_text.ArgumentWrapper(arguments=have_optional + have_required, is_group=True, sort_args=ai.sort_args)
            need_args = usage_text.ArgumentWrapper(arguments=need_required, is_group=True, sort_args=ai.sort_args)
            self._Error(parser_errors.ModalGroupError(parser=self, argument=usage_text.GetArgUsage(have_args, value=False, hidden=True, top=top), conflict=usage_text.GetArgUsage(need_args, value=False, hidden=True, top=top)))
    count = len(self.GetDestinations(have_required)) + len(self.GetDestinations(have_optional))
    if ai.is_mutex:
        conflict = usage_text.GetArgUsage(ai, value=False, hidden=True, top=top)
        if is_required and ai.is_required:
            if count != 1:
                if count:
                    argument = usage_text.GetArgUsage(sorted(have_required + have_optional, key=usage_text.GetArgSortKey)[0], value=False, hidden=True, top=top)
                    try:
                        flag = namespace.GetFlagArgument(argument)
                    except parser_errors.UnknownDestinationException:
                        flag = None
                    if flag:
                        value = namespace.GetValue(flag.dest)
                        if not isinstance(value, (bool, dict, list)):
                            argument = self._AddLocations(argument, value)
                else:
                    argument = None
                self._Error(parser_errors.RequiredMutexError(parser=self, argument=argument, conflict=conflict))
        elif count > 1:
            argument = usage_text.GetArgUsage(sorted(have_required + have_optional, key=usage_text.GetArgSortKey)[0], value=False, hidden=True, top=top)
            self._Error(parser_errors.OptionalMutexError(parser=self, argument=argument, conflict=conflict))
    return bool(count)