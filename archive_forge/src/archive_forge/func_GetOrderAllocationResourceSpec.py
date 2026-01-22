from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetOrderAllocationResourceSpec():
    return concepts.ResourceSpec('cloudcommerceconsumerprocurement.billingAccounts.orders.orderAllocations', resource_name='order-allocation', billingAccountsId=BillingAccountAttributeConfig(raw_help_text='Cloud Billing Account for the Procurement Order Allocation. Billing account id is required if order allocation is not specified as full resource name.'), ordersId=OrderAttributeConfig(raw_help_text='Procurement Order for the Order Allocation. Order id is required if order allocation is not specified as full resource name.'), orderAllocationsId=OrderAllocationAttributeConfig())