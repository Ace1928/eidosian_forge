from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.core import exceptions
def _ExtractMaskField(mask_path, api_field, is_dotted):
    """Extracts the api field name which constructs the mask used for request.

  For most update requests, you have to specify which fields in the resource
  should be updated. This information is stored as updateMask or fieldMask.
  Because resource and mask are in the same path level in a request, this
  function uses the mask_path as the guideline to extract the fields need to be
  parsed in the mask.

  Args:
    mask_path: string, the dotted path of mask in an api method, e.g. updateMask
      or updateRequest.fieldMask. The mask and the resource would always be in
      the same level in a request.
    api_field: string, the api field name in the resource to be updated and it
      is specified in the YAML files, e.g. displayName or
      updateRequest.instance.displayName.
    is_dotted: Boolean, True if the dotted path of the name is returned.

  Returns:
    String, the field name of the resource to be updated..

  """
    level = len(mask_path.split('.'))
    api_field_list = api_field.split('.')
    if is_dotted:
        if 'additionalProperties' in api_field_list:
            repeated_index = api_field_list.index('additionalProperties')
            api_field_list = api_field_list[:repeated_index]
        return '.'.join(api_field_list[level:])
    else:
        return api_field_list[level]