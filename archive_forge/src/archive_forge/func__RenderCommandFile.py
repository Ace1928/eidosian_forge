from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _RenderCommandFile(output_root, resource_data, collection_info, release_tracks, enable_overwrites):
    file_path = _BuildFilePath(output_root, _COMMAND_PATH_COMPONENTS, resource_data.home_directory, 'config', 'export.yaml')
    file_template = _BuildTemplate('command_template.tpl')
    context = _BuildCommandContext(collection_info, release_tracks, resource_data)
    _RenderFile(file_path, file_template, context, enable_overwrites)