from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
def _ShouldInvertBooleanFlag(self, name, action):
    """Checks if flag name with action is a Boolean flag to invert.

    Args:
      name: str, The flag name.
      action: argparse.Action, The argparse action.

    Returns:
      (False, None) if flag is not a Boolean flag or should not be inverted,
      (True, property) if flag is a Boolean flag associated with a property,
      (False, property) if flag is a non-Boolean flag associated with a property
      otherwise (True, None) if flag is a pure Boolean flag.
    """
    if not name.startswith('--'):
        return (False, None)
    if name.startswith('--no-'):
        return (False, None)
    if '--no-' + name[2:] in self.parser._option_string_actions:
        return (False, None)
    if _IsStoreBoolAction(action):
        return (True, None)
    prop, kind, _ = getattr(action, 'store_property', (None, None, None))
    if prop:
        return (kind == 'bool', prop)
    return (False, None)