from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
Updates a firewall rule for the given application.

    Args:
      resource: str, the resource path to the firewall rule.
      priority: int, the priority of the rule.
      source_range: str, optional ip address or range to take action on.
      action: firewall_rules_util.Action, optional action to take on matched
        addresses.
      description: str, optional string description of the rule.

    Returns:
      The updated firewall rule.

    Raises:
      NoFieldsSpecifiedError: when no fields have been specified for the update.
    