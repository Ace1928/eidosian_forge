import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_writing_completed(self):
    error = errors.WritingCompleted('a request')
    self.assertEqualDiff("The MediumRequest 'a request' has already had finish_writing called upon it - accept bytes may not be called anymore.", str(error))