from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _HandleDeadLetterPolicyUpdate(self, update_setting):
    if update_setting.value == CLEAR_DEAD_LETTER_VALUE:
        update_setting.value = None