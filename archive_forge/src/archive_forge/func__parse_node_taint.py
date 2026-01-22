from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_client as client
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
import six
def _parse_node_taint(self, node_taint):
    """Validates and parses a node taint object.

    Args:
      node_taint: tuple, of format (TAINT_KEY, value), where value is a string
        of format TAINT_VALUE:EFFECT.

    Returns:
      If taint is valid, returns a dict mapping message NodeTaint to its value;
      otherwise, raise ArgumentTypeError.
      For example,
      {
          'key': TAINT_KEY
          'value': TAINT_VALUE
          'effect': EFFECT
      }
    """
    taint_effect_enum = messages.NodeTaint.EffectValueValuesEnum
    taint_effect_mapping = {'NoSchedule': taint_effect_enum.NO_SCHEDULE, 'PreferNoSchedule': taint_effect_enum.PREFER_NO_SCHEDULE, 'NoExecute': taint_effect_enum.NO_EXECUTE}
    input_node_taint = '='.join(node_taint)
    valid_node_taint_effects = ', '.join((six.text_type(key) for key in sorted(taint_effect_mapping.keys())))
    taint_pattern = re.compile('([a-zA-Z0-9-_]*)=([a-zA-Z0-9-_]*):([a-zA-Z0-9-_]*)')
    taint_match = taint_pattern.fullmatch(input_node_taint)
    if not taint_match:
        raise arg_parsers.ArgumentTypeError('Node taint [{}] not in correct format, expect KEY=VALUE:EFFECT.'.format(input_node_taint))
    taint_key, taint_value, taint_effect = taint_match.groups()
    if taint_effect not in taint_effect_mapping:
        raise arg_parsers.ArgumentTypeError('Invalid taint effect in [{}] , expect one of [{}]'.format(input_node_taint, valid_node_taint_effects))
    return {'key': taint_key, 'value': taint_value, 'effect': taint_effect_mapping[taint_effect]}