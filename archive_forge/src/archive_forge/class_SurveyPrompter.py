from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import prompt_helper
class SurveyPrompter(prompt_helper.BasePrompter):
    """Manages prompting user for survey.

  Attributes:
     _prompt_record: PromptRecord, the record of the survey prompt history.
     _prompt_message: str, the prompting message.
  """
    _DEFAULT_SURVEY_PROMPT_MSG = 'To take a quick anonymous survey, run:\n  $ gcloud survey'

    def __init__(self, msg=_DEFAULT_SURVEY_PROMPT_MSG):
        self._prompt_record = PromptRecord()
        self._prompt_message = msg

    def PrintPromptMsg(self):
        log.status.write('\n\n' + self._prompt_message + '\n\n')

    def ShouldPrompt(self):
        """Check if the user should be prompted."""
        if not (log.out.isatty() and log.err.isatty()):
            return False
        last_prompt_time = self._prompt_record.last_prompt_time
        last_answer_survey_time = self._prompt_record.last_answer_survey_time
        now = time.time()
        if last_prompt_time and now - last_prompt_time < SURVEY_PROMPT_INTERVAL:
            return False
        if last_answer_survey_time and now - last_answer_survey_time < SURVEY_PROMPT_INTERVAL_AFTER_ANSWERED:
            return False
        return True

    def Prompt(self):
        """Prompts user for survey if user should be prompted."""
        if not self._prompt_record.CacheFileExists():
            with self._prompt_record as pr:
                pr.last_prompt_time = time.time()
            return
        if self.ShouldPrompt():
            self.PrintPromptMsg()
            with self._prompt_record as pr:
                pr.last_prompt_time = time.time()