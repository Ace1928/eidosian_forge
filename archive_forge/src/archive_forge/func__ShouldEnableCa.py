from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.privateca import create_utils
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import iam
from googlecloudsdk.command_lib.privateca import operations
from googlecloudsdk.command_lib.privateca import p4sa
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.privateca import storage
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _ShouldEnableCa(self, args, ca_ref):
    """Determines whether the CA should be enabled or not."""
    if args.auto_enable:
        return True
    ca_pool_name = ca_ref.Parent().RelativeName()
    list_response = self.client.projects_locations_caPools_certificateAuthorities.List(self.messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesListRequest(parent=ca_pool_name))
    if create_utils.HasEnabledCa(list_response.certificateAuthorities, self.messages):
        return False
    return console_io.PromptContinue(message='The CaPool [{}] has no enabled CAs and cannot issue any certificates until at least one CA is enabled. Would you like to also enable this CA?'.format(ca_ref.Parent().Name()), default=False)