from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.command_lib.survey import util as survey_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
class RatingQuestion(Question):
    """"Rating question.

  Attributes:
     min_answer: int, minimum acceptable value for answer.
     max_answer: int, maximum acceptable value for answer.
  """

    @classmethod
    def FromDictionary(cls, content):
        try:
            return cls(**content)
        except TypeError:
            raise QuestionCreationError(required_fields=['question', 'instruction', 'instruction_on_rejection', 'min_answer', 'max_answer'])

    def __init__(self, question, instruction, instruction_on_rejection, min_answer, max_answer, answer=None):
        super(RatingQuestion, self).__init__(question=question, instruction=instruction, instruction_on_rejection=instruction_on_rejection, answer=answer)
        self._min = min_answer
        self._max = max_answer

    def _PrintQuestion(self):
        question = survey_util.Indent(self._question, 1)
        log.Print(question)

    def AcceptAnswer(self, answer):
        try:
            answer_int = int(answer)
            return self._min <= answer_int <= self._max
        except ValueError:
            return False

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._question == other._question and self._instruction == other._instruction and (self._instruction_on_rejection == other._instruction_on_rejection) and (self._min == other._min) and (self._max == other._max)
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self._question, self._instruction, self._instruction_on_rejection, self._min, self._max))