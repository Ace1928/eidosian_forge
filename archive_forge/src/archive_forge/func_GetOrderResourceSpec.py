from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetOrderResourceSpec():
    return concepts.ResourceSpec('cloudcommerceconsumerprocurement.billingAccounts.orders', resource_name='order', billingAccountsId=BillingAccountAttributeConfig(raw_help_text='Cloud Billing Account for the Procurement Order. Billing account id is required if order is not specified as full resource name.'), ordersId=OrderAttributeConfig())