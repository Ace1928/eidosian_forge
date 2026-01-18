from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
def validateFlags(self, args):
    if not args.no_service_account and (not args.service_account):
        raise exceptions.RequiredArgumentException('--service-account', 'must be specified with a service account. To clear the default service account use [--no-service-account].')