from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
import ruamel.yaml as ryaml
def ParseAndValidateMessage(input_str):
    """Validate that yaml string or object is a valid OutputMessage."""
    try:
        yaml_object = yaml.load(input_str)
        _MSG_VALIDATOR.Validate(yaml_object)
        time_util.Strptime(yaml_object['timestamp'])
        resources = yaml_object.get('resource_body')
        if resources and (not IsResourceLike(resources)):
            raise ValueError(_INVALID_RESOURCE_VALUE_MSG)
        return yaml_object
    except (yaml.YAMLParseError, ValueError) as e:
        raise MessageParsingError('Error loading YAML message [{}] :: {}.'.format(input_str, e))
    except (yaml_validator.ValidationError, ryaml.error.YAMLStreamError) as ve:
        raise InvalidMessageError('Invalid OutputMessage string [{}] :: [{}]'.format(input_str, ve))