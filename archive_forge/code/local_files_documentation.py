from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import itertools
from googlecloudsdk.command_lib.code import builders
from googlecloudsdk.command_lib.code import local
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.core import yaml
import six
Create a skaffold yaml file.

    Args:
      kubernetes_file_path: Path to the kubernetes config file.

    Returns:
      Text of the skaffold yaml file.
    