from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.command_lib.survey import util as survey_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
Returns True if answer is accepted, otherwise returns False.

    Accepts any answer for free text question.

    Args:
      answer: str, the answer to check.

    Returns:
       True
    