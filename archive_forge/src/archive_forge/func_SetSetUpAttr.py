import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
@staticmethod
def SetSetUpAttr(cls, test_names):
    """Wraps setUp() with per-class setUp() functionality."""
    cls_setUp = cls.setUp

    def setUp(self):
        """Function that will encapsulate and replace cls.setUp()."""
        leaf = self.__class__
        if leaf.__tests_to_run is None:
            leaf.__tests_to_run = set(test_names)
            self.setUpTestCase()
        cls_setUp(self)
    BeforeAfterTestCaseMeta.SetMethod(cls, 'setUp', setUp)