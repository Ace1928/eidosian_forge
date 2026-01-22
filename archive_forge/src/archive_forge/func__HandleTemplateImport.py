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
def _HandleTemplateImport(import_object):
    """Takes a template and looks for its schema to process.

  Args:
    import_object: Template object whose schema to check for and process

  Returns:
    List of import_objects that the schema is importing.

  Raises:
    ConfigError: If any of the schema's imported items are missing the
        'path' field.
  """
    schema_path = import_object.GetFullPath() + '.schema'
    schema_name = import_object.GetName() + '.schema'
    schema_object = _BuildFileImportObject(schema_path, schema_name)
    if not schema_object.Exists():
        return []
    import_objects = _GetImportObjects(schema_object)
    import_objects.append(schema_object)
    return import_objects