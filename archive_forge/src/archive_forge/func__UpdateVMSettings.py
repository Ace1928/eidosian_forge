from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import appinfo_includes
from googlecloudsdk.third_party.appengine.api import croninfo
from googlecloudsdk.third_party.appengine.api import dispatchinfo
from googlecloudsdk.third_party.appengine.api import queueinfo
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def _UpdateVMSettings(self):
    """Overwrites vm_settings for App Engine services with VMs.

    Also sets module_yaml_path which is needed for some runtimes.

    Raises:
      AppConfigError: if the function was called for a standard service
    """
    if self.env not in [env.MANAGED_VMS, env.FLEX]:
        raise AppConfigError('This is not an App Engine Flexible service. Please set `env` field to `flex`.')
    if not self.parsed.vm_settings:
        self.parsed.vm_settings = appinfo.VmSettings()
    self.parsed.vm_settings['module_yaml_path'] = os.path.basename(self.file)