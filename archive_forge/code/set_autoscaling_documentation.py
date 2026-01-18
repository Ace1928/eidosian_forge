from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import autoscalers as autoscalers_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
Set autoscaling parameters of a managed instance group.