from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
def _BuildImportObject(config=None, template=None, composite_type=None, properties=None):
    """Build an import object from the given config name."""
    if composite_type:
        if not _IsValidCompositeTypeSyntax(composite_type):
            raise exceptions.ConfigError('Invalid composite type syntax.')
        return _ImportSyntheticCompositeTypeFile(composite_type, properties)
    if config:
        return _BuildFileImportObject(config)
    if template:
        return _BuildFileImportObject(template)
    raise exceptions.ConfigError('No path or name for a config, template, or composite type was specified.')