from typing import Union, IO, Any
from io import StringIO
import sys
from .ruamel_yaml import YAML
from .ruamel_yaml.representer import RepresenterError
from .util import force_path, FilePath, YAMLInput, YAMLOutput
def yaml_dumps(data: YAMLInput, indent_mapping: int=2, indent_sequence: int=4, indent_offset: int=2, sort_keys: bool=False) -> str:
    """Serialize an object to a YAML string. See the ruamel.yaml docs on
    indentation for more details on the expected format.
    https://yaml.readthedocs.io/en/latest/detail.html?highlight=indentation#indentation-of-block-sequences

    data: The YAML-serializable data.
    indent_mapping (int): Mapping indentation.
    indent_sequence (int): Sequence indentation.
    indent_offset (int): Indentation offset.
    sort_keys (bool): Sort dictionary keys.
    RETURNS (str): The serialized string.
    """
    yaml = CustomYaml()
    yaml.sort_base_mapping_type_on_output = sort_keys
    yaml.indent(mapping=indent_mapping, sequence=indent_sequence, offset=indent_offset)
    return yaml.dump(data)