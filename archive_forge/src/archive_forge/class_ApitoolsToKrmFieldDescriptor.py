from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import os
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
class ApitoolsToKrmFieldDescriptor(object):
    """Ecapsulates the mapping from an apitools message field to a YAML path.

  Attributes:
    message_field: string, The field in the apitools message.
    yaml_path: string, Dot ('.') seperated path to the correlated field data in
      the yaml input file.
    submessage_template: {string: ApitoolsToKrmFieldDescriptor}, dict of
      ApitoolsToKrmFieldDescriptors describing the template of the submessage.
      None if this descriptor describes a scalar field.
    repeatable: bool, True if this desriptor is for a repeatable field,
      False otherwise.
  """

    def __init__(self, message_field, yaml_field_path, submessage_template=None, repeatable=False):
        self._message_path = message_field
        self._yaml_path = yaml_field_path
        self._submessage_template = submessage_template
        self._repeatable = repeatable

    @property
    def message_field(self):
        return self._message_path

    @property
    def yaml_path(self):
        return self._yaml_path

    @property
    def submessage_template(self):
        return self._submessage_template

    @property
    def repeatable(self):
        return self._repeatable

    def __str__(self):
        output = collections.OrderedDict()
        output[self._message_path] = self._yaml_path
        output['repeatable'] = self._repeatable
        submessage_template_str_array = []
        if self._submessage_template:
            for descriptor in self._submessage_template.values():
                submessage_template_str_array.append(str(descriptor))
        output['submessage_template'] = submessage_template_str_array or None
        yaml.convert_to_block_text(output)
        return yaml.dump(output, round_trip=True)

    def __eq__(self, other):
        if not isinstance(other, ApitoolsToKrmFieldDescriptor):
            return False
        return self._message_path == other.message_field and self._yaml_path == other.yaml_path and (self._submessage_template == other.submessage_template) and (self._repeatable == other.repeatable)

    def __hash__(self):
        return hash((self._message_path, self._yaml_path, self._repeatable, self.__str__()))

    @classmethod
    def FromYamlData(cls, msg_field, data):
        """Construct ApitoolsToKrmFieldDescriptor from a string or a dict."""
        msg_field = msg_field.strip()
        if isinstance(data, six.string_types):
            return cls(message_field=msg_field, yaml_field_path=data.strip())
        elif isinstance(data, dict):
            submsg_data = data.get('submessage_template')
            if submsg_data:
                submessage_template = collections.OrderedDict([(f, cls.FromYamlData(f, v)) for f, v in six.iteritems(submsg_data)])
            else:
                submessage_template = None
            return cls(message_field=msg_field, yaml_field_path=data['yaml_path'].strip(), repeatable=data.get('repeatable', False), submessage_template=submessage_template)
        else:
            raise ValueError('Can not parse ApitoolsToKrmFieldDescriptor for [{}] from data: [{}]'.format(msg_field, data))