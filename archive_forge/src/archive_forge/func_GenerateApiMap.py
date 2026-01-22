from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import os
from apitools.gen import gen_client
from googlecloudsdk.api_lib.regen import api_def
from googlecloudsdk.api_lib.regen import resource_generator
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
import six
def GenerateApiMap(base_dir, root_dir, api_config):
    """Create an apis_map.py file in the given root_dir with for given api_config.

  Args:
      base_dir: str, Path of directory for the project.
      root_dir: str, Path of the map file location within the project.
      api_config: regeneration config for all apis.
  """
    api_def_filename, _ = os.path.splitext(api_def.__file__)
    api_def_source = files.ReadFileContents(api_def_filename + '.py')
    tpl = template.Template(filename=os.path.join(os.path.dirname(__file__), 'template.tpl'))
    api_map_file = os.path.join(base_dir, root_dir, 'apis_map.py')
    logging.debug('Generating api map at %s', api_map_file)
    api_map = _MakeApiMap(root_dir.replace('/', '.'), api_config)
    logging.debug('Creating following api map %s', api_map)
    with files.FileWriter(api_map_file) as apis_map_file:
        ctx = runtime.Context(apis_map_file, api_def_source=api_def_source, apis_map=api_map)
        tpl.render_context(ctx)