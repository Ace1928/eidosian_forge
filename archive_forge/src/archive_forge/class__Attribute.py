from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.calliope.concepts import deps_map_util
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class _Attribute(object):
    """A base class for concept attributes.

  Attributes:
    name: The name of the attribute. Used primarily to control the arg or flag
      name corresponding to the attribute. Must be in all lower case.
    param_name: corresponds to where the attribute is mapped in the resource
    help_text: String describing the attribute's relationship to the concept,
      used to generate help for an attribute flag.
    required: True if the attribute is required.
    fallthroughs: [googlecloudsdk.calliope.concepts.deps_lib.Fallthrough], the
      list of sources of data, in priority order, that can provide a value for
      the attribute if not given on the command line. These should only be
      sources inherent to the attribute, such as associated properties, not
      command-specific sources.
    completer: core.cache.completion_cache.Completer, the completer associated
      with the attribute.
    value_type: the type to be accepted by the attribute arg. Defaults to str.
  """

    def __init__(self, name, param_name, help_text=None, required=False, fallthroughs=None, completer=None, value_type=None):
        """Initializes."""
        if re.search('[A-Z]', name) and re.search('r[a-z]', name):
            raise ValueError('Invalid attribute name [{}]: Attribute names should be in lower snake case (foo_bar) so they can be transformed to flag names.'.format(name))
        self.name = name
        self.param_name = param_name or name
        self.help_text = help_text
        self.required = required
        self.fallthroughs = fallthroughs or []
        self.completer = completer
        self.value_type = value_type or str

    def __eq__(self, other):
        """Overrides."""
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name and self.param_name == other.param_name and (self.help_text == other.help_text) and (self.required == other.required) and (self.completer == other.completer) and (self.fallthroughs == other.fallthroughs) and (self.value_type == other.value_type)

    def __hash__(self):
        return sum(map(hash, [self.name, self.param_name, self.help_text, self.required, self.completer, self.value_type])) + sum(map(hash, self.fallthroughs))