from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _FormatCheckList(check_list):
    buf = io.StringIO()
    for check in check_list:
        usage_text.WrapWithPrefix(check.name, check.description, 20, 78, '  ', writer=buf)
    return buf.getvalue()