import sys
import threading
import unittest
import gi
from gi.repository import GObject, Gtk    # noqa: E402
from testtools import StreamToExtendedDecorator  # noqa: E402
from subunit import (PROGRESS_POP, PROGRESS_PUSH, PROGRESS_SET,  # noqa: E402
from subunit.progress_model import ProgressModel  # noqa: E402
def update_counts(self):
    self.run_label.set_text(str(self.testsRun))
    bad = len(self.failures + self.errors)
    self.ok_label.set_text(str(self.testsRun - bad))
    self.not_ok_label.set_text(str(bad))