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
class ExternalRuntimeConfigurator(Configurator):
    """Configurator for general externalized runtimes.

  Attributes:
    runtime: (ExternalizedRuntime) The runtime that produced this.
    params: (Params) Runtime parameters.
    data: ({str: object, ...} or None) Optional dictionary of runtime data
      passed back through a runtime_parameters message.
    generated_appinfo: ({str: object, ...} or None) Generated appinfo if any
      is produced by the runtime.
    path: (str) Path to the user's source directory.
  """

    def __init__(self, runtime, params, data, generated_appinfo, path, env):
        """Constructor.

    Args:
      runtime: (ExternalizedRuntime) The runtime that produced this.
      params: (Params) Runtime parameters.
      data: ({str: object, ...} or None) Optional dictionary of runtime data
        passed back through a runtime_parameters message.
      generated_appinfo: ({str: object, ...} or None) Optional dictionary
        representing the contents of app.yaml if the runtime produces this.
      path: (str) Path to the user's source directory.
      env: (ExecutionEnvironment)
    """
        self.runtime = runtime
        self.params = params
        self.data = data
        if generated_appinfo:
            self.generated_appinfo = {}
            if not 'env' in generated_appinfo:
                self.generated_appinfo['env'] = 'flex'
            self.generated_appinfo.update(generated_appinfo)
        else:
            self.generated_appinfo = None
        self.path = path
        self.env = env

    def MaybeWriteAppYaml(self):
        """Generates the app.yaml file if it doesn't already exist."""
        if not self.generated_appinfo:
            return
        notify = logging.info if self.params.deploy else self.env.Print
        filename = os.path.join(self.path, 'app.yaml')
        if self.params.appinfo or os.path.exists(filename):
            notify(FILE_EXISTS_MESSAGE.format('app.yaml'))
            return
        notify(WRITING_FILE_MESSAGE.format('app.yaml', self.path))
        with open(filename, 'w') as f:
            yaml.safe_dump(self.generated_appinfo, f, default_flow_style=False)

    def SetGeneratedAppInfo(self, generated_appinfo):
        """Sets the generated appinfo."""
        self.generated_appinfo = generated_appinfo

    def CollectData(self):
        self.runtime.CollectData(self)

    def Prebuild(self):
        self.runtime.Prebuild(self)

    def GenerateConfigs(self):
        self.MaybeWriteAppYaml()
        if not self.params.appinfo and self.generated_appinfo:
            self.params.appinfo = comm.dict_to_object(self.generated_appinfo)
        return self.runtime.GenerateConfigs(self)

    def GenerateConfigData(self):
        self.MaybeWriteAppYaml()
        if not self.params.appinfo and self.generated_appinfo:
            self.params.appinfo = comm.dict_to_object(self.generated_appinfo)
        return self.runtime.GenerateConfigData(self)