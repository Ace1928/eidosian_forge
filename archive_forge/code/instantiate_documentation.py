from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.core import log
Instantiate a workflow template.