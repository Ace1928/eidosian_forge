from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
def GetRuntimes(args):
    """Gets a list of unique runtimes that the user is about to run.

  Args:
    args: A list of arguments (typically sys.argv).

  Returns:
    A set of runtime strings. If python27 and libraries section is populated
    in any of the yaml-files, 'python27-libs', a fake runtime id, will be part
    of the set, in conjunction with the original 'python27'.

  Raises:
    MultipleAppYamlError: The supplied application configuration has duplicate
      app yamls.
  """
    runtimes = set()
    for arg in args:
        yaml_candidate = None
        if os.path.isfile(arg) and os.path.splitext(arg)[1] in _YAML_FILE_EXTENSIONS:
            yaml_candidate = arg
        elif os.path.isdir(arg):
            for extension in _YAML_FILE_EXTENSIONS:
                fullname = os.path.join(arg, 'app' + extension)
                if os.path.isfile(fullname):
                    if yaml_candidate:
                        raise MultipleAppYamlError('Directory "{0}" contains conflicting files {1}'.format(arg, ' and '.join(yaml_candidate)))
                    yaml_candidate = fullname
        if yaml_candidate:
            try:
                info = yaml.load_path(yaml_candidate)
            except yaml.Error:
                continue
            if not isinstance(info, dict):
                continue
            if 'runtime' in info:
                runtime = info.get('runtime')
                if type(runtime) == str:
                    if runtime == 'python27' and info.get('libraries'):
                        runtimes.add('python27-libs')
                    runtimes.add(runtime)
                    if runtime in _WARNING_RUNTIMES:
                        log.warning(_WARNING_RUNTIMES[runtime])
        elif os.path.isfile(os.path.join(arg, 'WEB-INF', 'appengine-web.xml')):
            runtimes.add('java')
    return runtimes