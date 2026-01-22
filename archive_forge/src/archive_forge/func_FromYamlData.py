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