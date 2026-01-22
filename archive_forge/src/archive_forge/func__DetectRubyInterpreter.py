from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _DetectRubyInterpreter(path, bundler_available):
    """Determines the ruby interpreter and version expected by this application.

  Args:
    path: (str) Application path.
    bundler_available: (bool) Whether bundler is available in the environment.

  Returns:
    (str or None) The interpreter version in rbenv (.ruby-version) format, or
    None to use the base image default.
  """
    if bundler_available:
        ruby_info = _RunSubprocess('bundle platform --ruby')
        if not re.match('^No ', ruby_info):
            match = re.match('^ruby (\\d+\\.\\d+(\\.\\d+)?)', ruby_info)
            if match:
                ruby_version = match.group(1)
                ruby_version = RUBY_VERSION_MAP.get(ruby_version, ruby_version)
                msg = '\nUsing Ruby {0} as requested in the Gemfile.'.format(ruby_version)
                log.status.Print(msg)
                return ruby_version
            msg = 'Unrecognized platform in Gemfile: [{0}]'.format(ruby_info)
            log.status.Print(msg)
    ruby_version = _ReadFile(path, '.ruby-version')
    if ruby_version:
        ruby_version = ruby_version.strip()
        msg = '\nUsing Ruby {0} as requested in the .ruby-version file'.format(ruby_version)
        log.status.Print(msg)
        return ruby_version
    msg = '\nNOTICE: We will deploy your application using a recent version of the standard "MRI" Ruby runtime by default. If you want to use a specific Ruby runtime, you can create a ".ruby-version" file in this directory. (For best performance, we recommend MRI version {0}.)'.format(PREFERRED_RUBY_VERSION)
    log.status.Print(msg)
    return None