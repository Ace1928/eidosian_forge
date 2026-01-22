from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeRegionalPublicDelegatedPrefixesArg():
    return compute_flags.ResourceArgument(resource_name='public delegated prefix', regional_collection='compute.publicDelegatedPrefixes')