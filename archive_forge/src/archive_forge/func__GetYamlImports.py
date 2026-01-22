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
def _GetYamlImports(import_object, globbing_enabled=False):
    """Extract the import section of a file.

  If the glob_imports config is set to true, expand any globs (e.g. *.jinja).
  Named imports cannot be used with globs that expand to more than one file.
  If globbing is disabled or a glob pattern does not expand to match any files,
  importer will use the literal string as the file path.

  Args:
    import_object: The object in which to look for imports.
    globbing_enabled: If true, will resolved glob patterns dynamically.

  Returns:
    A list of dictionary objects, containing the keys 'path' and 'name' for each
    file to import. If no name was found, we populate it with the value of path.

  Raises:
   ConfigError: If we cannont read the file, the yaml is malformed, or
       the import object does not contain a 'path' field.
  """
    parent_dir = None
    if not _IsUrl(import_object.full_path):
        parent_dir = os.path.dirname(os.path.abspath(import_object.full_path))
    content = import_object.GetContent()
    yaml_content = yaml.load(content)
    imports = []
    if yaml_content and IMPORTS in yaml_content:
        raw_imports = yaml_content[IMPORTS]
        for i in raw_imports:
            if PATH not in i:
                raise exceptions.ConfigError('Missing required field %s in import in file %s.' % (PATH, import_object.full_path))
            glob_matches = []
            if globbing_enabled and parent_dir and (not _IsUrl(i[PATH])):
                with files.ChDir(parent_dir):
                    glob_matches = glob.glob(i[PATH])
                    glob_matches = _SanitizeWindowsPathsGlobs(glob_matches)
                if len(glob_matches) > 1:
                    if NAME in i:
                        raise exceptions.ConfigError('Cannot use import name %s for path glob in file %s that matches multiple objects.' % (i[NAME], import_object.full_path))
                    imports.extend([{NAME: g, PATH: g} for g in glob_matches])
                    continue
            if len(glob_matches) == 1:
                i[PATH] = glob_matches[0]
            if NAME not in i:
                i[NAME] = i[PATH]
            imports.append(i)
    return imports