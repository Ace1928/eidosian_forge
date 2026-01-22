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
Loads the YAML data from the given reference.

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
      