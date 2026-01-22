from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def UpdateThresholdRules(ref, args, req):
    """Add threshold rule to budget."""
    messages = GetMessagesModule(args)
    version = GetApiVersion(args)
    client = apis.GetClientInstance('billingbudgets', version)
    budgets = client.billingAccounts_budgets
    get_request_type = messages.BillingbudgetsBillingAccountsBudgetsGetRequest
    get_request = get_request_type(name=six.text_type(ref.RelativeName()))
    old_threshold_rules = budgets.Get(get_request).thresholdRules
    if args.IsSpecified('clear_threshold_rules'):
        old_threshold_rules = []
        GetVersionedUpdateBillingBudget(args, req).thresholdRules = old_threshold_rules
    if args.IsSpecified('add_threshold_rule'):
        added_threshold_rules = args.add_threshold_rule
        final_rules = AddRules(old_threshold_rules, added_threshold_rules)
        GetVersionedUpdateBillingBudget(args, req).thresholdRules = final_rules
        return req
    if args.IsSpecified('threshold_rules_from_file'):
        rules_from_file = yaml.load(args.threshold_rules_from_file)
        if version == 'v1':
            budget = messages_util.DictToMessageWithErrorCheck({'thresholdRules': rules_from_file}, messages.GoogleCloudBillingBudgetsV1Budget)
            req.googleCloudBillingBudgetsV1Budget.updateMask += ',thresholdRules'
        else:
            budget = messages_util.DictToMessageWithErrorCheck({'thresholdRules': rules_from_file}, messages.GoogleCloudBillingBudgetsV1beta1Budget)
        req.googleCloudBillingBudgetsV1beta1UpdateBudgetRequest.updateMask += ',thresholdRules'
        GetVersionedUpdateBillingBudget(args, req).thresholdRules = budget.thresholdRules
    return req