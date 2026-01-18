import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_set_application_data(self):
    """
        set application data

        """
    old = self.ETSConfig.application_data
    self.ETSConfig.application_data = 'foo'
    self.assertEqual('foo', self.ETSConfig.application_data)
    self.ETSConfig.application_data = old
    self.assertEqual(old, self.ETSConfig.application_data)