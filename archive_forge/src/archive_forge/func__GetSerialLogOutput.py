from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.api_lib.logging import util as logging_util
from googlecloudsdk.core import log as logging
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _GetSerialLogOutput(client, project, instance, zone, port):
    request = (client.apitools_client.instances, 'GetSerialPortOutput', client.messages.ComputeInstancesGetSerialPortOutputRequest(instance=instance.name, project=project.name, port=port, start=0, zone=zone))
    return client.MakeRequests(requests=[request])[0]