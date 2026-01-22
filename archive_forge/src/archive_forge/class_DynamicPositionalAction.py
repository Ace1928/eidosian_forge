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
class DynamicPositionalAction(six.with_metaclass(abc.ABCMeta, CloudSDKSubParsersAction)):
    """An argparse action that adds new flags to the parser when it is called.

  We need to use a subparser for this because for a given parser, argparse
  collects all the arg information before it starts parsing. Adding in new flags
  on the fly doesn't work. With a subparser, it is independent so we can load
  flags into here on the fly before argparse loads this particular parser.
  """

    def __init__(self, *args, **kwargs):
        self.hidden = kwargs.pop('hidden', False)
        self._parent_ai = kwargs.pop('parent_ai')
        super(DynamicPositionalAction, self).__init__(*args, **kwargs)

    def IsValidChoice(self, choice):
        self._AddParser(choice)
        return True

    def LoadAllChoices(self):
        pass

    def _AddParser(self, choice):
        return self.add_parser(choice, add_help=False, prog=self._parent_ai.parser.prog, calliope_command=self._parent_ai.parser._calliope_command)

    @abc.abstractmethod
    def GenerateArgs(self, namespace, choice):
        pass

    @abc.abstractmethod
    def Completions(self, prefix, parsed_args, **kwargs):
        pass

    def __call__(self, parser, namespace, values, option_string=None):
        choice = values[0]
        args = self.GenerateArgs(namespace, choice)
        sub_parser = self._name_parser_map[choice]
        ai = parser_arguments.ArgumentInterceptor(sub_parser, is_global=False, cli_generator=None, allow_positional=True, data=self._parent_ai.data)
        for flag in itertools.chain(self._parent_ai.flag_args, self._parent_ai.ancestor_flag_args):
            if flag.do_not_propagate or flag.is_required:
                continue
            sub_parser._add_action(flag)
        ai.display_info.AddLowerDisplayInfo(self._parent_ai.display_info)
        for arg in args:
            arg.RemoveFromParser(ai)
            added_arg = arg.AddToParser(ai)
            if '_ARGCOMPLETE' in os.environ and (not hasattr(added_arg, '_orig_class')):
                added_arg._orig_class = added_arg.__class__
        super(DynamicPositionalAction, self).__call__(parser, namespace, values, option_string=option_string)
        if '_ARGCOMPLETE' not in os.environ:
            self._name_parser_map.clear()