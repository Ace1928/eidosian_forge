import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_set_company(self):
    """
        set company

        """
    old = self.ETSConfig.company
    self.ETSConfig.company = 'foo'
    self.assertEqual('foo', self.ETSConfig.company)
    self.ETSConfig.company = old
    self.assertEqual(old, self.ETSConfig.company)