from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
Updates an application.

    Args:
      split_health_checks: Boolean, whether to enable split health checks by
        default.
      service_account: str, the app-level default service account to update for
        this App Engine app.

    Returns:
      Long running operation.
    