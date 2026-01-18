import os
import sys
import breezy
from breezy import osutils, trace
from breezy.tests import (TestCase, TestCaseInTempDir, TestSkipped,
Check output from version command and master option is reasonable