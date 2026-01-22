from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.calliope.exceptions import core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
from mako import runtime
from mako import template
def GenerateImportScript(import_data, dest_file=None, dest_dir=None):
    """Generate Terraform import shell script from template.

  Args:
    import_data: string, Import data for each resource.
    dest_file: string, Filename path to write the generated script to. If
      dest_file is None, then a default filename will be generated.
    dest_dir: string, Directory path to write the generated script to. If
      dest_dir is None, then script will be written to CWD.

  Returns:
    tuple(string, int, [string])), the path to the generated script, number of
      import statements generated and list of files that could not be processed.

  Raises:
    TerraformGenerationError: If and error occurs writing to disk/stdout.
  """
    output_file_name = os.path.join(dest_dir, dest_file)
    context = {'data': []}
    for import_path, import_statement in import_data:
        _, module_name = ConstructModuleParameters(import_path, dest_dir)
        import_cmd_data = import_statement.partition(_IMPORT_CMD_PREFIX)[1:]
        context['data'].append('{cmd} module.{module_name}.{cmd_sfx}'.format(cmd=import_cmd_data[0], module_name=module_name, cmd_sfx=import_cmd_data[1].strip()))
    context['data'] = os.linesep.join(context['data'])
    output_template = None
    template_key = 'WINDOWS' if platforms.OperatingSystem.IsWindows() else 'BASH'
    if template_key == 'WINDOWS':
        output_template = _BuildTemplate('windows_shell_template.tpl')
    elif template_key == 'BASH':
        context['bash_comments'] = _BASH_COMMENTS
        output_template = _BuildTemplate('bash_shell_template.tpl')
    try:
        with files.FileWriter(output_file_name, create_path=True) as f:
            ctx = runtime.Context(f, **context)
            output_template.render_context(ctx)
        os.chmod(output_file_name, 493)
    except files.Error as e:
        raise TerraformGenerationError('Error writing import script::{}'.format(e))
    return (output_file_name, len(import_data))