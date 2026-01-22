from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetOrderOperationResourceSpec():
    return concepts.ResourceSpec('cloudcommerceconsumerprocurement.billingAccounts.orders.operations', resource_name='order-operation', billingAccountsId=BillingAccountAttributeConfig(name='order-operation-billing-account'), ordersId=OrderAttributeConfig(name='order-operation-order'), operationsId=OperationAttributeConfig())