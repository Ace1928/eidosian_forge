from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import log
import six
def _CreateRequest(args, messages):
    """_CreateRequest creates CreateServiceAccountIdentityBindingRequests."""
    req = messages.CreateServiceAccountIdentityBindingRequest(acceptanceFilter=args.acceptance_filter, cel=_EncodeAttributeTranslatorCEL(args.attribute_translator_cel, messages), oidc=messages.IDPReferenceOIDC(audience=args.oidc_audience, maxTokenLifetimeSeconds=args.oidc_max_token_lifetime, url=args.oidc_issuer_url))
    return messages.IamProjectsServiceAccountsIdentityBindingsCreateRequest(createServiceAccountIdentityBindingRequest=req, name=iam_util.EmailToAccountResourceName(args.service_account))