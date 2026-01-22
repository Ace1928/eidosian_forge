import typing
from typing import Mapping, Tuple
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import yaml
Reads Secret object from yaml file.

  Args:
    secret_file_path: path to the file.

  Returns:
    tuple with name and data of the Secret.

  Raises:
    command_util.InvalidUserInputError: if the content of the file is invalid.
  