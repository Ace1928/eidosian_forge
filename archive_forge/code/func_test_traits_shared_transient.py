import os
import sys
import tempfile
import textwrap
import shutil
import subprocess
import unittest
from traits.api import (
from traits.testing.optional_dependencies import requires_numpy
def test_traits_shared_transient(self):

    class LazyProperty(TraitType):
        default_value_type = DefaultValue.constant

        def get(self, obj, name):
            return 1729
    self.assertFalse(Float().transient)
    LazyProperty().as_ctrait()
    self.assertFalse(Float().transient)