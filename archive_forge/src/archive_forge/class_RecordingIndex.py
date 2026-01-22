import re
import textwrap
from .. import (builtins, commands, config, errors, help, help_topics, i18n,
from .test_i18n import ZzzTranslations
class RecordingIndex:

    def __init__(self, name):
        self.prefix = name

    def get_topics(self, topic):
        calls.append(('get_topics', self.prefix, topic))
        return ['something']