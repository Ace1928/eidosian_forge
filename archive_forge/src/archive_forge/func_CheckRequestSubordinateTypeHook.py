from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core.util import times
def CheckRequestSubordinateTypeHook(version='v1'):
    """Raises an exception if the request is not for a subordinate ca."""

    def CheckRequestSubordinateTypeHookVersioned(resource_ref, unused_args, request):
        _CheckRequestTypeHook(resource_ref, base.GetMessagesModule(api_version=version).CertificateAuthority.TypeValueValuesEnum.SUBORDINATE)
        return request
    return CheckRequestSubordinateTypeHookVersioned