from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
def _AddBetaFlags(self):
    """Set up flags that are for alpha and beta tracks."""
    self.BuildersGroup().AddDockerfile()
    self.AddSource()
    self.AddLocalPort()
    self.CredentialsGroup().AddServiceAccount()
    self.CredentialsGroup().AddApplicationDefaultCredential()
    self.AddReadinessProbe()
    self.AddAllowSecretManagerFlag()
    self.AddSecrets()
    self.BuildersGroup().AddBuilder()