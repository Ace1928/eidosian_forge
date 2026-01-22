from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.identity import admin_directory
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def ChoiceToEnum(choice, enum_type, item_type='choice', valid_choices=None):
    """Converts the typed choice into an apitools Enum value."""
    if choice is None:
        return None
    name = ChoiceToEnumName(choice)
    valid_choices = valid_choices or [arg_utils.EnumNameToChoice(n) for n in enum_type.names()]
    try:
        return enum_type.lookup_by_name(name)
    except KeyError:
        raise arg_parsers.ArgumentTypeError('Invalid {item}: {selection}. Valid choices are: [{values}].'.format(item=item_type, selection=arg_utils.EnumNameToChoice(name), values=', '.join((c for c in sorted(valid_choices)))))