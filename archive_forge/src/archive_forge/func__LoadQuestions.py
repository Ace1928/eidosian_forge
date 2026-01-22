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
def _LoadQuestions(self):
    """Generator of questions in this survey."""
    for q in self._survey_content['questions']:
        question_type = q['question_type']
        if not hasattr(question, question_type):
            raise QuestionTypeNotDefinedError('The question type is not defined.')
        yield getattr(question, question_type).FromDictionary(q['properties'])