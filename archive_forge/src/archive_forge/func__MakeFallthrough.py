from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def _MakeFallthrough(self, fallthrough_string):
    """Make an ArgFallthrough from a formatted string."""
    values = fallthrough_string.split('.')
    if len(values) == 1:
        arg_name = values
        return deps.ArgFallthrough(values[0])
    elif len(values) == 2:
        spec_name, attribute_name = values
        spec = self.specs.get(spec_name)
        arg_name = spec.attribute_to_args_map.get(attribute_name, None)
        if not arg_name:
            raise ValueError('Invalid fallthrough value [{}]: No argument associated with attribute [{}] in concept argument named [{}]'.format(fallthrough_string, attribute_name, spec_name))
        return deps.ArgFallthrough(arg_name)
    else:
        raise ValueError('bad fallthrough string [{}]'.format(fallthrough_string))