from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddCreateTagTemplateFlags():
    """Hook for adding flags to tag-template create."""
    field_flag = base.Argument('--field', type=arg_parsers.ArgDict(spec={'id': str, 'type': str, 'display-name': str, 'required': bool}, required_keys=['id', 'type']), action='append', required=True, metavar='id=ID,type=TYPE,display-name=DISPLAY_NAME,required=REQUIRED', help="        Specification for a tag template field. This flag can be repeated to\n        specify multiple fields. The following keys are allowed:\n\n          *id*::: (Required) ID of the tag template field.\n\n          *type*::: (Required) Type of the tag template field. Choices are\n              double, string, bool, timestamp, and enum.\n\n                    To specify a string field:\n                      `type=string`\n\n                    To specify an enum field with values 'A' and 'B':\n                      `type=enum(A|B)`\n\n          *display-name*::: Display name of the tag template field.\n\n          *required*::: Indicates if the tag template field is required.\n              Defaults to FALSE.\n      ")
    return [field_flag]