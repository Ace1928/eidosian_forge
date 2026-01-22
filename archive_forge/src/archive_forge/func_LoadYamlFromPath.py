from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def LoadYamlFromPath(path):
    try:
        data = yaml.load_path(path, round_trip=True, preserve_quotes=True)
    except yaml.Error as e:
        raise cloudbuild_exceptions.ParserError(path, e.inner_error)
    if not yaml.dict_like(data):
        raise cloudbuild_exceptions.ParserError(path, 'Could not parse as a dictionary.')
    return data