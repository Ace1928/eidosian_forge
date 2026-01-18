import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_pickle_unpickle(self):
    stream = io.StringIO('foo')
    runner = unittest.TextTestRunner(stream)
    for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
        s = pickle.dumps(runner, protocol)
        obj = pickle.loads(s)
        self.assertEqual(obj.stream.getvalue(), stream.getvalue())