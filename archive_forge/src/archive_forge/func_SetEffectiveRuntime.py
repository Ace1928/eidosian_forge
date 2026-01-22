from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def SetEffectiveRuntime(self, runtime):
    """Sets the runtime while respecting vm runtimes rules for runtime settings.

    Args:
       runtime: The runtime to use.
    """
    if self.IsVm():
        if not self.vm_settings:
            self.vm_settings = VmSettings()
        self.vm_settings['vm_runtime'] = runtime
        self.runtime = 'vm'
    else:
        self.runtime = runtime