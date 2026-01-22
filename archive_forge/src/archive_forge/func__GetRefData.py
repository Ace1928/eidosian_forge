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
def _GetRefData(self, path):
    """Loads the YAML data from the given reference.

      A YAML reference must refer to a YAML file and an attribute within that
      file to extract.

      Args:
        path: str, The path of the YAML file to import. It must be in the
          form of: package.module:attribute.attribute, where the module path is
          separated from the sub attributes within the YAML by a ':'.

      Raises:
        LayoutException: If the given module or attribute cannot be loaded.

      Returns:
        The referenced YAML data.
      """
    parts = path.split(':')
    if len(parts) != 2:
        raise LayoutException('Invalid Yaml reference: [{}]. References must be in the format: path(.path)+:attribute(.attribute)*'.format(path))
    path_segments = parts[0].split('.')
    try:
        root_module = importlib.import_module(path_segments[0])
        yaml_path = os.path.join(os.path.dirname(root_module.__file__), *path_segments[1:]) + '.yaml'
        data = _SafeLoadYamlFile(yaml_path)
    except (ImportError, IOError) as e:
        raise LayoutException('Failed to load Yaml reference file [{}]: {}'.format(parts[0], e))
    return self._GetAttribute(data, parts[1], yaml_path)