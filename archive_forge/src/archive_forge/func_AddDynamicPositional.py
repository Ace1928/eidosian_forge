from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
def AddDynamicPositional(self, name, action, **kwargs):
    """Add a positional argument that adds new args on the fly when called.

    Args:
      name: The name/dest of the positional argument.
      action: The argparse Action to use. It must be a subclass of
        parser_extensions.DynamicPositionalAction.
      **kwargs: Passed verbatim to the argparse.ArgumentParser.add_subparsers
        method.

    Returns:
      argparse.Action, The added action.
    """
    kwargs['dest'] = name
    if 'metavar' not in kwargs:
        kwargs['metavar'] = name.upper()
    kwargs['parent_ai'] = self
    action = self.parser.add_subparsers(action=action, **kwargs)
    action.completer = action.Completions
    action.is_group = False
    action.is_hidden = kwargs.get('hidden', False)
    action.is_positional = True
    action.is_required = True
    self.positional_args.append(action)
    self.arguments.append(action)
    return action