import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_reading_completed(self):
    error = errors.ReadingCompleted('a request')
    self.assertEqualDiff("The MediumRequest 'a request' has already had finish_reading called upon it - the request has been completed and no more data may be read.", str(error))