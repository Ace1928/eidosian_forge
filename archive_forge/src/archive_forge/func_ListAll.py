from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def ListAll(directory=None):
    """Returns the CliTreeInfo list of all available CLI trees.

  Args:
    directory: The config directory containing the CLI tree modules.

  Raises:
    CliTreeVersionError: loaded tree version mismatch
    ImportModuleError: import errors

  Returns:
    The CLI tree.
  """
    directories = [directory, cli_tree.CliTreeConfigDir(), cli_tree.CliTreeDir()]
    trees = []
    for directory in directories:
        if not directory or not os.path.exists(directory):
            continue
        for dirpath, _, filenames in os.walk(six.text_type(directory)):
            for filename in sorted(filenames):
                base, extension = os.path.splitext(filename)
                if base == '__init__' or '.' in base:
                    continue
                path = os.path.join(dirpath, filename)
                error = ''
                tree = None
                if extension in ('.py', '.pyc'):
                    try:
                        module = module_util.ImportPath(path)
                    except module_util.ImportModuleError as e:
                        error = six.text_type(e)
                    try:
                        tree = module.TREE
                    except AttributeError:
                        tree = None
                elif extension == '.json':
                    try:
                        tree = json.loads(files.ReadFileContents(path))
                    except Exception as e:
                        error = six.text_type(e)
                if tree:
                    version = tree.get(cli_tree.LOOKUP_VERSION, 'UNKNOWN')
                    cli_version = tree.get(cli_tree.LOOKUP_CLI_VERSION, 'UNKNOWN')
                    del tree
                else:
                    version = 'UNKNOWN'
                    cli_version = 'UNKNOWN'
                trees.append(CliTreeInfo(command=base, path=_ParameterizePath(path), version=version, cli_version=cli_version, command_installed=bool(files.FindExecutableOnPath(base)), error=error))
            break
    return trees