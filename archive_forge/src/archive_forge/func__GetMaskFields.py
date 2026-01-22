from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.core import exceptions
def _GetMaskFields(param, args, mask_path, is_dotted):
    """Gets the fieldMask based on the yaml arg and the arguments specified.

  Args:
    param: yaml_arg_schema.YAMLArgument, the yaml argument added to parser
    args: parser_extensions.Namespace, user specified arguments
    mask_path: str, path to where update mask applies
    is_dotted: bool, True if the dotted path of the name is returned

  Returns:
    Set of fields (str) to add to the update mask
  """
    field_set = set()
    if not param.IsApiFieldSpecified(args):
        return field_set
    for api_field in param.api_fields:
        mask_field = _ExtractMaskField(mask_path, api_field, is_dotted)
        if mask_field:
            field_set.add(mask_field)
    return field_set