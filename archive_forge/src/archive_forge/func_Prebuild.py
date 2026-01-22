from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import subprocess
import sys
import threading
from . import comm
import ruamel.yaml as yaml
from six.moves import input
def Prebuild(self, configurator):
    """Perform any additional build behavior before the application is deployed.

    Args:
      configurator: (ExternalRuntimeConfigurator) The configurator returned by
      Detect().
    """
    prebuild = self.config.get('prebuild')
    if prebuild:
        result = self.RunPlugin('prebuild', prebuild, configurator.params, args=[configurator.path], runtime_data=configurator.data)
        if result.docker_context:
            configurator.path = result.docker_context