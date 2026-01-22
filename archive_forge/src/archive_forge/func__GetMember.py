from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import json
import os
import pipes
import re
import shlex
import sys
import types
from fire import completion
from fire import decorators
from fire import formatting
from fire import helptext
from fire import inspectutils
from fire import interact
from fire import parser
from fire import trace
from fire import value_types
from fire.console import console_io
import six
def _GetMember(component, args):
    """Returns a subcomponent of component by consuming an arg from args.

  Given a starting component and args, this function gets a member from that
  component, consuming one arg in the process.

  Args:
    component: The component from which to get a member.
    args: Args from which to consume in the search for the next component.
  Returns:
    component: The component that was found by consuming an arg.
    consumed_args: The args that were consumed by getting this member.
    remaining_args: The remaining args that haven't been consumed yet.
  Raises:
    FireError: If we cannot consume an argument to get a member.
  """
    members = dir(component)
    arg = args[0]
    arg_names = [arg, arg.replace('-', '_')]
    for arg_name in arg_names:
        if arg_name in members:
            return (getattr(component, arg_name), [arg], args[1:])
    raise FireError('Could not consume arg:', arg)