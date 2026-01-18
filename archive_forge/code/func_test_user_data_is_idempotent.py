import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_user_data_is_idempotent(self):
    """
        user data is idempotent

        """
    self.test_user_data()