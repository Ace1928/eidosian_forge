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
def EnsureSameSetName(self, set_name, named_set):
    if set_name is not None and hasattr(named_set, 'name'):
        if set_name != named_set.name:
            raise exceptions.BadArgumentException('set-name', 'The set name provided [{0}] does not match the one from the file [{1}]'.format(set_name, named_set.name))
    if not hasattr(named_set, 'name') and set_name is not None:
        named_set.name = set_name