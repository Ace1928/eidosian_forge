from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _PreActionHook(action, func, additional_help=None):
    """Allows an function hook to be injected before an Action executes.

  Wraps an Action in another action that can execute an arbitrary function on
  the argument value before passing invocation to underlying action.
  This is useful for:
  - Chaining actions together at runtime.
  - Adding additional pre-processing or logging to an argument/flag
  - Adding instrumentation to runtime execution of an flag without changing the
  underlying intended behavior of the flag itself

  Args:
    action: action class to be wrapped. Either a subclass of argparse.Action
        or a string representing one of the built in arg_parse action types.
        If None, argparse._StoreAction type is used as default.
    func: callable, function to be executed before invoking the __call__ method
        of the wrapped action. Takes value from command line.
    additional_help: _AdditionalHelp, Additional help (label, message) to be
        added to action help

  Returns:
    argparse.Action, wrapper action to use.

  Raises:
    TypeError: If action or func are invalid types.
  """
    if not callable(func):
        raise TypeError('func should be a callable of the form func(value)')
    if not isinstance(action, six.string_types) and (not issubclass(action, argparse.Action)):
        raise TypeError('action should be either a subclass of argparse.Action or a string representing one of the default argparse Action Types')

    class Action(argparse.Action):
        """Action Wrapper Class."""
        wrapped_action = action

        @classmethod
        def SetWrappedAction(cls, action):
            cls.wrapped_action = action

        def _GetActionClass(self):
            if isinstance(self.wrapped_action, six.string_types):
                action_cls = GetArgparseBuiltInAction(self.wrapped_action)
            else:
                action_cls = self.wrapped_action
            return action_cls

        def __init__(self, *args, **kwargs):
            if additional_help:
                original_help = kwargs.get('help', '').rstrip()
                kwargs['help'] = '{0} {1}\n+\n{2}'.format(additional_help.label, original_help, additional_help.message)
            self._wrapped_action = self._GetActionClass()(*args, **kwargs)
            self.func = func
            kwargs['nargs'] = self._wrapped_action.nargs
            kwargs['const'] = self._wrapped_action.const
            kwargs['choices'] = self._wrapped_action.choices
            kwargs['option_strings'] = self._wrapped_action.option_strings
            super(Action, self).__init__(*args, **kwargs)

        def __call__(self, parser, namespace, value, option_string=None):
            flag_value = getattr(namespace, self.dest, None)
            if isinstance(flag_value, list):
                if len(flag_value) < 1:
                    self.func(value)
            elif not value:
                self.func(self._wrapped_action.const)
            else:
                self.func(value)
            self._wrapped_action(parser, namespace, value, option_string)
    return Action