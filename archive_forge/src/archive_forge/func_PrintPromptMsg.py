from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import prompt_helper
def PrintPromptMsg(self):
    log.status.write('\n\n' + self._prompt_message + '\n\n')