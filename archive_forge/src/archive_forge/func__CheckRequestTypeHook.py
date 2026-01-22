from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core.util import times
def _CheckRequestTypeHook(resource_ref, expected_type, version='v1'):
    """Do a get on a CA resource and check its type against expected_type."""
    client = base.GetClientInstance(api_version=version)
    messages = base.GetMessagesModule(api_version=version)
    certificate_authority = client.projects_locations_caPools_certificateAuthorities.Get(messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesGetRequest(name=resource_ref.RelativeName()))
    resource_args.CheckExpectedCAType(expected_type, certificate_authority)