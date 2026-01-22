from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import textwrap
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core.util import files
import six
def RunPack(name, builder, runtime, entry_point, path, build_env_vars):
    """Runs Pack Build with the command built from arguments of the command parser.

  Args:
    name: Name of the image build.
    builder: Name of the builder by the flag.
    runtime: Runtime specified by flag.
    entry_point: Entry point of the function specified by flag.
    path: Source of the zip file.
    build_env_vars: Build environment variables.
  Raises:
    PackExecutionException: if the exit code of the execution is non-zero.
  """
    pack_cmd = [_PACK, 'build', '--builder']
    if not builder:
        [language, version] = re.findall('(\\D+|\\d+)', runtime)
        if int(version) >= _RUNTIME_MINVERSION_UBUNTU_22[language]:
            builder = _GOOGLE_22_BUILDER if language == 'dotnet' else _APPENGINE_BUILDER.format(22, language)
        else:
            builder = _V1_BUILDER if language == 'dotnet' else _APPENGINE_BUILDER.format(18, language)
    pack_cmd.append(builder)
    if build_env_vars:
        _AddEnvVars(pack_cmd, build_env_vars)
    pack_cmd.extend(['--env', 'GOOGLE_FUNCTION_TARGET=' + entry_point])
    pack_cmd.extend(['--path', path])
    pack_cmd.extend(['-q', name])
    status = execution_utils.Exec(pack_cmd, no_exit=True)
    if status:
        raise PackExecutionException(status, 'Pack failed to build the container image.')