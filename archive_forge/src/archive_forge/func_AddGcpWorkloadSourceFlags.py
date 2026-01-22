from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import Collection
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
def AddGcpWorkloadSourceFlags(parser):
    parser.add_argument('--resources', type=arg_parsers.ArgList(), help='A list of allowed resources for the workload source.', metavar='RESOURCE')
    parser.add_argument('--attached-service-accounts', type=arg_parsers.ArgList(), help='A list of allowed attached_service_accounts for the workload source.', metavar='SERVICE_ACCOUNT')