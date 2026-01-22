from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.declarative import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _GetBillingParams(self, args_namspace):
    """Process billing project flags in args and return Terraform settings."""
    use_gcloud_billing = args_namspace.use_gcloud_billing_project
    user_project_override = billing_project = None
    if use_gcloud_billing:
        billing_project = properties.VALUES.billing.quota_project.Get()
        user_project_override = 'true'
    elif args_namspace.tf_user_project_override:
        billing_project = args_namspace.tf_billing_project
        user_project_override = 'true'
    return (user_project_override, billing_project)