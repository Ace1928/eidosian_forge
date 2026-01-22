import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class OldTextResult(ResultWithNoStartTestRunStopTestRun):
    separator2 = ''

    def printErrors(self):
        pass