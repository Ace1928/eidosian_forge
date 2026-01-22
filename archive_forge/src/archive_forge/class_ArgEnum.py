from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.calliope import arg_parsers
class ArgEnum(object):
    """Interpret an argument value as an item from an allowed value list.

  Example usage:

    parser.add_argument(
      '--agent-rules',
      metavar='type=TYPE,version=VERSION,package-state=PACKAGE-STATE,
               enable-autoupgrade=ENABLE-AUTOUPGRADE',
      action='store',
      required=True,
      type=arg_parsers.ArgList(
          custom_delim_char=';',
          element_type=arg_parsers.ArgDict(spec={
              'type': ArgEnum('type', [OpsAgentPolicy.AgentRule.Type]),
              'version': str,
              'package_state': str,
              'enable_autoupgrade': arg_parsers.ArgBoolean(),
          }),
      )
    )

  Example error:

    ERROR: (gcloud.alpha.compute.instances.ops-agents.policies.create) argument
    --agent-rules: Invalid value [what] from field [type], expected one of
    [logging,
    metrics].
  """

    def __init__(self, field_name, allowed_values):
        """Constructor.

    Args:
      field_name: str. The name of field that contains this arg value.
      allowed_values: list of allowed values. The allowed values to validate
        against.
    """
        self._field_name = field_name
        self._allowed_values = allowed_values

    def __call__(self, arg_value):
        """Interpret the arg value as an item from an allowed value list.

    Args:
      arg_value: str. The value of the user input argument.

    Returns:
      The value of the arg.

    Raises:
      arg_parsers.ArgumentTypeError.
        If the arg value is not one of the allowed values.
    """
        str_value = str(arg_value)
        if str_value not in self._allowed_values:
            raise arg_parsers.ArgumentTypeError('Invalid value [{0}] from field [{1}], expected one of [{2}].'.format(arg_value, self._field_name, ', '.join(self._allowed_values)))
        return str_value