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
def UpdateParseToMoneyTypeV1Beta1(money):
    """Convert the input to Money Type for v1beta1 Update method."""
    messages = GetMessagesModuleForVersion('v1beta1')
    return ParseMoney(money, messages)