from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.code import kubernetes
from googlecloudsdk.command_lib.run import secrets_mapping
def IsValidK8sName(name):
    return re.match('[a-z0-9]([a-z0-9\\-\\.]{0,251}[a-z0-9])?$', name)