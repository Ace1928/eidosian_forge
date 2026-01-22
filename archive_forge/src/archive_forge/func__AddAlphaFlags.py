from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def _AddAlphaFlags(self):
    """Set up flags that are for alpha track only."""
    self.AddCloudsqlInstances()
    self.AddServiceName()
    self.AddImage()
    self.AddMemory()
    self.AddCpu()
    self.EnvVarsGroup().AddEnvVars()
    self.EnvVarsGroup().AddEnvVarsFile()
    self.AddCloud()