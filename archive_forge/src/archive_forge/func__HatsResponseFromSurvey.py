from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import platform
import socket
import time
from googlecloudsdk.command_lib.survey import question
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.survey import survey_check
from googlecloudsdk.core.util import platforms
from six.moves import http_client as httplib
def _HatsResponseFromSurvey(survey_instance):
    """Puts survey response to a HatsResponse object.

  Args:
    survey_instance: googlecloudsdk.command_lib.survey.survey.Survey, a survey
      object which contains user's response.

  Returns:
    HatsResponse as a dictionary to send to concord.
  """
    hats_metadata = {'site_id': 'CloudSDK', 'site_name': 'googlecloudsdk', 'survey_id': survey_instance.name}
    multi_choice_questions = []
    rating_questions = []
    open_text_questions = []
    for i, q in enumerate(survey_instance):
        if not q.IsAnswered():
            continue
        if isinstance(q, question.MultiChoiceQuestion):
            answer_int = len(q) + 1 - int(q.answer)
            multi_choice_questions.append({'question_number': i, 'order_index': [answer_int], 'answer_index': [answer_int], 'answer_text': [q.Choice(int(q.answer))], 'order': list(range(1, len(q) + 1))})
        elif isinstance(q, question.RatingQuestion):
            rating_questions.append({'question_number': i, 'rating': int(q.answer)})
        elif isinstance(q, question.FreeTextQuestion):
            open_text_questions.append({'question_number': i, 'answer_text': q.answer})
    response = {'hats_metadata': hats_metadata}
    if multi_choice_questions:
        response['multiple_choice_response'] = multi_choice_questions
    if rating_questions:
        response['rating_response'] = rating_questions
    if open_text_questions:
        response['open_text_response'] = open_text_questions
    return response