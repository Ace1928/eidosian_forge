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
def CollectData(self, configurator):
    """Do data collection on a detected runtime.

    Args:
      configurator: (ExternalRuntimeConfigurator) The configurator retuned by
        Detect().

    Raises:
      InvalidRuntimeDefinition: For a variety of problems with the runtime
        definition.
    """
    collect_data = self.config.get('collectData')
    if collect_data:
        result = self.RunPlugin('collect_data', collect_data, configurator.params, runtime_data=configurator.data)
        if result.generated_appinfo:
            configurator.SetGeneratedAppInfo(result.generated_appinfo)