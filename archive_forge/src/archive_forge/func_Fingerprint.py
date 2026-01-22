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
def Fingerprint(path, params):
    """Check for a Ruby app.

  Args:
    path: (str) Application path.
    params: (ext_runtime.Params) Parameters passed through to the
      fingerprinters.

  Returns:
    (RubyConfigurator or None) Returns a configurator if the path contains a
    Ruby app, or None if not.
  """
    appinfo = params.appinfo
    if not _CheckForRubyRuntime(path, appinfo):
        return None
    bundler_available = _CheckEnvironment(path)
    gems = _DetectGems(bundler_available)
    ruby_version = _DetectRubyInterpreter(path, bundler_available)
    packages = _DetectNeededPackages(gems)
    if appinfo and appinfo.entrypoint:
        entrypoint = appinfo.entrypoint
    else:
        default_entrypoint = _DetectDefaultEntrypoint(path, gems)
        entrypoint = _ChooseEntrypoint(default_entrypoint, appinfo)
    return RubyConfigurator(path, params, ruby_version, entrypoint, packages)