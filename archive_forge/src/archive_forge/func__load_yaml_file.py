import os
from vitrageclient.common import yaml_utils
from vitrageclient import exceptions as exc
@classmethod
def _load_yaml_file(cls, path):
    with open(path, 'r') as stream:
        return cls._load_yaml(stream)