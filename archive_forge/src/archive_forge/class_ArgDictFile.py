from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
class ArgDictFile(object):
    """Interpret a YAML file as a dict."""

    def __init__(self, key_type=None, value_type=None):
        """Initialize an ArgDictFile.

    Args:
      key_type: (str)->str, A function to apply to each of the dict keys.
      value_type: (str)->str, A function to apply to each of the dict values.
    """
        self.key_type = key_type
        self.value_type = value_type

    def __call__(self, file_path):
        map_file_dict = yaml.load_path(file_path)
        map_dict = {}
        if not yaml.dict_like(map_file_dict):
            raise arg_parsers.ArgumentTypeError('Invalid YAML/JSON data in [{}], expected map-like data.'.format(file_path))
        for key, value in map_file_dict.items():
            if self.key_type:
                try:
                    key = self.key_type(key)
                except ValueError:
                    raise arg_parsers.ArgumentTypeError('Invalid key [{0}]'.format(key))
            if self.value_type:
                try:
                    value = self.value_type(value)
                except ValueError:
                    raise arg_parsers.ArgumentTypeError('Invalid value [{0}]'.format(value))
            map_dict[key] = value
        return map_dict