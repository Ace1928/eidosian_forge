from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
import shutil
import googlecloudsdk
from googlecloudsdk import third_party
from googlecloudsdk.api_lib.regen import generate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.meta import regen as regen_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import ruamel.yaml
import six
from six.moves import map
def _LoadConfig(config_file_name=None):
    """Loads regen config from given filename."""
    config_file_name = config_file_name or os.path.join(os.path.dirname(encoding.Decode(third_party.__file__)), 'regen_apis_config.yaml')
    if not os.path.isfile(config_file_name):
        raise regen_utils.ConfigFileError('{} Not found'.format(config_file_name))
    with files.FileReader(config_file_name) as stream:
        config = ruamel.yaml.round_trip_load(stream)
    if not config or 'root_dir' not in config:
        raise regen_utils.ConfigFileError('{} does not have format of gcloud api config file'.format(config_file_name))
    return config