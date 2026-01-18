import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,

        Check that warnings argument of TextTestRunner correctly affects the
        behavior of the warnings.
        