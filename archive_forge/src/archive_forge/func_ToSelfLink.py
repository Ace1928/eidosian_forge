from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.billing import billing_client
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
@staticmethod
def ToSelfLink(account):
    return resources.REGISTRY.Parse(account.name, collection='cloudbilling.billingAccounts').SelfLink()