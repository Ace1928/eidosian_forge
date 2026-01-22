from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base as calliope_base
import googlecloudsdk.generated_clients.apis.telcoautomation.v1.telcoautomation_v1_messages as GAConfig
import googlecloudsdk.generated_clients.apis.telcoautomation.v1alpha1.telcoautomation_v1alpha1_messages as AlphaConfig
def GetVersionedMasterAuthorizedNetworksConfig(args):
    version = GetApiVersion(args)
    if version == 'v1alpha1':
        return AlphaConfig.MasterAuthorizedNetworksConfig()
    else:
        return GAConfig.MasterAuthorizedNetworksConfig()