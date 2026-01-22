from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import enum
from googlecloudsdk.command_lib.survey import question
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import pkg_resources
def IsSatisfied(self):
    """Returns if survey respondent is satisfied."""
    satisfaction_question = self.questions[0]
    if satisfaction_question.IsAnswered():
        return satisfaction_question.IsSatisfied()
    else:
        return None