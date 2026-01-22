from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as functions_api_util
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddQuotaValidationFlag(parser, hidden=False):
    """Add --quota-validation flag."""
    parser.add_argument('--quota-validation', hidden=hidden, help='Input to control quota checks for resources in terraform configuration files. There are limited resources on which quota validation applies. Supported values are QUOTA_VALIDATION_UNSPECIFIED, ENABLED, ENFORCED', type=QuotaValidationEnum)