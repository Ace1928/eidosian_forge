from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core.util import times
def CheckResponseSubordinateTypeHookVersioned(response, unused_args):
    resource_args.CheckExpectedCAType(base.GetMessagesModule(api_version=version).CertificateAuthority.TypeValueValuesEnum.SUBORDINATE, response, version=version)
    return response