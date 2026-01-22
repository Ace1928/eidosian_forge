from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import retry
import six
def CreateAutoscalingPolicy(dataproc, name, policy):
    """Returns the server-resolved policy after creating the given policy.

  Args:
    dataproc: wrapper for dataproc resources, client and messages.
    name: The autoscaling policy resource name.
    policy: The AutoscalingPolicy message to create.
  """
    parent = '/'.join(name.split('/')[0:4])
    request = dataproc.messages.DataprocProjectsRegionsAutoscalingPoliciesCreateRequest(parent=parent, autoscalingPolicy=policy)
    policy = dataproc.client.projects_regions_autoscalingPolicies.Create(request)
    log.status.Print('Created [{0}].'.format(policy.id))
    return policy