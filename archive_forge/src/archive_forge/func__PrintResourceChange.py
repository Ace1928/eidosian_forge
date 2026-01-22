from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from collections import OrderedDict
import contextlib
import copy
import datetime
import json
import logging
import os
import sys
import time
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import parser as style_parser
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def _PrintResourceChange(operation, resource, kind, is_async, details, failed, operation_past_tense=None):
    """Prints a status message for operation on resource.

  The non-failure status messages are disabled when user output is disabled.

  Args:
    operation: str, The completed operation name.
    resource: str, The resource name.
    kind: str, The resource kind (instance, cluster, project, etc.).
    is_async: bool, True if the operation is in progress.
    details: str, Extra details appended to the message. Keep it succinct.
    failed: str, Failure message. For commands that operate on multiple
      resources and report all successes and failures before exiting. Failure
      messages use log.error. This will display the message on the standard
      error even when user output is disabled.
    operation_past_tense: str, The past tense version of the operation verb.
      If None assumes operation + 'd'
  """
    msg = []
    if failed:
        msg.append('Failed to ')
        msg.append(operation)
    elif is_async:
        msg.append(operation.capitalize())
        msg.append(' in progress for')
    else:
        verb = operation_past_tense or '{0}d'.format(operation)
        msg.append('{0}'.format(verb.capitalize()))
    if kind:
        msg.append(' ')
        msg.append(kind)
    if resource:
        msg.append(' ')
        msg.append(text.TextTypes.RESOURCE_NAME(six.text_type(resource)))
    if details:
        msg.append(' ')
        msg.append(details)
    if failed:
        msg.append(': ')
        msg.append(failed)
    period = '' if str(msg[-1]).endswith('.') else '.'
    msg.append(period)
    msg = text.TypedText(msg)
    writer = error if failed else status.Print
    writer(msg)