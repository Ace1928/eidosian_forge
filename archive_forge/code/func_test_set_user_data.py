import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_set_user_data(self):
    """
        set user data

        """
    old = self.ETSConfig.user_data
    self.ETSConfig.user_data = 'foo'
    self.assertEqual('foo', self.ETSConfig.user_data)
    self.ETSConfig.user_data = old
    self.assertEqual(old, self.ETSConfig.user_data)