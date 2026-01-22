from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import importlib
import os
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_release_tracks
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import pkg_resources
from ruamel import yaml
import six
def _ConstructMappingHelper(self, macro, source_func, data):
    attribute_path = data.pop(macro, None)
    if not attribute_path:
        return data
    modified_data = {}
    for path in attribute_path.split(','):
        modified_data.update(source_func(path))
    modified_data.update(data)
    return modified_data