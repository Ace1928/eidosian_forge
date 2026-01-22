import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def assertCompletionContains(self, *words):
    missing = set(words) - self.completion_result
    if missing:
        raise AssertionError('Completion should contain %r but it has %r' % (missing, self.completion_result))