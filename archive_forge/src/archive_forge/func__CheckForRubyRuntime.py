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
def _CheckForRubyRuntime(path, appinfo):
    """Determines whether to treat this application as runtime:ruby.

  Honors the appinfo runtime setting; otherwise looks at the contents of the
  current directory and confirms with the user.

  Args:
    path: (str) Application path.
    appinfo: (apphosting.api.appinfo.AppInfoExternal or None) The parsed
      app.yaml file for the module if it exists.

  Returns:
    (bool) Whether this app should be treated as runtime:ruby.
  """
    if appinfo and appinfo.GetEffectiveRuntime() == 'ruby':
        return True
    log.info('Checking for Ruby.')
    gemfile_path = os.path.join(path, 'Gemfile')
    if not os.path.isfile(gemfile_path):
        return False
    got_ruby_message = 'This looks like a Ruby application.'
    if console_io.CanPrompt():
        return console_io.PromptContinue(message=got_ruby_message, prompt_string='Proceed to configure deployment for Ruby?')
    else:
        log.info(got_ruby_message)
        return True