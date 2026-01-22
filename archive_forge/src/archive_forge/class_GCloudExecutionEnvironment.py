from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from gae_ext_runtime import ext_runtime
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
class GCloudExecutionEnvironment(ext_runtime.ExecutionEnvironment):
    """ExecutionEnvironment implemented using gcloud's core functions."""

    def GetPythonExecutable(self):
        return execution_utils.GetPythonExecutable()

    def CanPrompt(self):
        return console_io.CanPrompt()

    def PromptResponse(self, message):
        return console_io.PromptResponse(message)

    def Print(self, message):
        return log.status.Print(message)