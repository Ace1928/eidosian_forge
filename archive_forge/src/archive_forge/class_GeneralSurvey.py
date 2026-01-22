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
class GeneralSurvey(Survey):
    """GeneralSurvey defined in googlecloudsdk/command_lib/survey/contents."""
    SURVEY_NAME = 'GeneralSurvey'

    def __init__(self):
        super(GeneralSurvey, self).__init__(self.SURVEY_NAME)

    def __iter__(self):
        yield self.questions[0]
        yield self.questions[1]
        if self.IsSatisfied() is None or self.IsSatisfied():
            yield self.questions[2]
        else:
            yield self.questions[3]

    def IsSatisfied(self):
        """Returns if survey respondent is satisfied."""
        satisfaction_question = self.questions[0]
        if satisfaction_question.IsAnswered():
            return satisfaction_question.IsSatisfied()
        else:
            return None