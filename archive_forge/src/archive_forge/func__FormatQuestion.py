from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.command_lib.survey import util as survey_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
import six
def _FormatQuestion(self, indexes):
    """Formats question to present to users."""
    choices_repr = ['[{}] {}'.format(index, msg) for index, msg in zip(indexes, self._choices)]
    choices_repr = [survey_util.Indent(content, 2) for content in choices_repr]
    choices_repr = '\n'.join(choices_repr)
    question_repr = survey_util.Indent(self._question, 1)
    return '\n'.join([question_repr, choices_repr])