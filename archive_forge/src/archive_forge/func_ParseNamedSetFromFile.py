from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import os
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def ParseNamedSetFromFile(self, input_file, file_format, messages):
    resource = self.ParseFile(input_file, file_format)
    if 'resource' in resource:
        resource = resource['resource']
    named_set = messages_util.DictToMessageWithErrorCheck(resource, messages.NamedSet)
    if 'fingerprint' in resource:
        named_set.fingerprint = base64.b64decode(resource['fingerprint'])
    return named_set