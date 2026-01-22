from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os.path
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def WriteYaml(command_tpl_name, collection_dict, output_dir, api_message_module):
    """Writes command's YAML file; returns True if file written, else False.

  Args:
    command_tpl_name: name of command template file
    collection_dict: a mapping of collection info to feed template
    output_dir: path to directory in which to write YAML file. If command YAML
    file already exists in this location, the user will be prompted to
    choose to override it or not.
    api_message_module: the API's message module, used to check if command
    type is supported by API
  Returns:
    True if declarative file is written, False if user chooses not to
    override an existing file OR API does not support command type, and no
    new file is written.
  """
    command_name = command_tpl_name[:-len(TEMPLATE_SUFFIX)]
    command_name_capitalized = ''.join([word.capitalize() for word in command_name.split('_')])
    if command_name == 'describe':
        command_name_capitalized = 'Get'
    collection_prefix = ''.join([_GetResourceMessageClassName(word) for word in collection_dict['collection_name'].split('.')])
    expected_message_name = collection_prefix + command_name_capitalized + 'Request'
    alt_create_message_name = collection_prefix + 'InsertRequest'
    command_supported = False
    for message_name in dir(api_message_module):
        if message_name == expected_message_name or message_name == alt_create_message_name:
            command_supported = True
    command_yaml_tpl = _TemplateFileForCommandPath(command_tpl_name)
    command_filename = command_name + '.yaml'
    full_command_path = os.path.join(output_dir, command_filename)
    file_already_exists = os.path.exists(full_command_path)
    overwrite = False
    if file_already_exists:
        overwrite = console_io.PromptContinue(default=False, throw_if_unattended=True, message='{command_filename} already exists, and continuing will overwrite the old file. The scenario test skeleton file for this command will only be generated if you continue'.format(command_filename=command_filename))
    if (not file_already_exists or overwrite) and command_supported:
        with files.FileWriter(full_command_path) as f:
            ctx = runtime.Context(f, **collection_dict)
            command_yaml_tpl.render_context(ctx)
        log.status.Print('New file written at ' + full_command_path)
        return True
    else:
        log.status.Print('No new file written at ' + full_command_path)
        return False