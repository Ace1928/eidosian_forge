import typing
from typing import Mapping, Tuple
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import yaml
def _ReadConfigMapFromFile(config_map_file_path: str) -> Tuple[str, Mapping[str, str]]:
    """Reads ConfigMap object from yaml file.

  Args:
    config_map_file_path: path to the file.

  Returns:
    tuple with name and data of the ConfigMap.

  Raises:
    command_util.InvalidUserInputError: if the content of the file is invalid.
  """
    config_map_file_content = yaml.load_path(config_map_file_path)
    if not isinstance(config_map_file_content, dict):
        raise command_util.InvalidUserInputError(f'Invalid content of the {config_map_file_path}')
    kind = config_map_file_content.get('kind')
    metadata_name = config_map_file_content.get('metadata', {}).get('name', '')
    data = config_map_file_content.get('data', {})
    if kind != 'ConfigMap':
        raise command_util.InvalidUserInputError(f'Incorrect "kind" attribute value. Found: {kind}, should be: ConfigMap')
    if not metadata_name:
        raise command_util.InvalidUserInputError(f'Empty metadata.name in {config_map_file_path}')
    return (metadata_name, data)