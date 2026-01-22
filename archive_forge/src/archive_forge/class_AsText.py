import sys
from testtools import TestResult
from testtools.content import StackLinesContent
from testtools.matchers import (
from testtools import runtest
class AsText(AfterPreprocessing):
    """Match the text of a Content instance."""

    def __init__(self, matcher, annotate=True):
        super().__init__(lambda log: log.as_text(), matcher, annotate=annotate)