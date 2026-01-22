from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
import itertools
import re
import uuid
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
@staticmethod
def IsRemotePath(secret_name):
    return bool(re.search(ReachableSecret._REMOTE_SECRET_FLAG_VALUE, secret_name))