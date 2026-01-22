from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
def GetFlag(self, parameter_name, parameter_value=None, check_properties=True, for_update=False):
    """Returns the command line flag for parameter.

    If the flag is already present in program values, returns None.
    If the user needs to specify it, returns a string in the form
    '--flag-name=value'. If the flag is boolean and True, returns '--flag-name'.

    Args:
      parameter_name: The parameter name.
      parameter_value: The parameter value if not None. Otherwise
        GetValue() is used to get the value.
      check_properties: Check property values if parsed_args don't help.
      for_update: Return flag for a cache update command.

    Returns:
      The command line flag  for the parameter, or None.
    """
    del for_update
    attribute_name = self._AttributeName(parameter_name)
    flag_name = self.resource_info.attribute_to_args_map.get(attribute_name, None)
    if not flag_name:
        if attribute_name == 'project':
            flag_name = '--project'
        else:
            return None
    program_value = self.GetValue(parameter_name)
    if parameter_value != program_value:
        if parameter_value is None:
            parameter_value = program_value
        if parameter_value:
            if parameter_value is True:
                return flag_name
            return '{name}={value}'.format(name=flag_name, value=parameter_value)
    return None