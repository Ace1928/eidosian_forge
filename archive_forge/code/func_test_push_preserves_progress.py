import unittest
from subunit.progress_model import ProgressModel
def test_push_preserves_progress(self):
    progress = ProgressModel()
    progress.adjust_width(3)
    progress.advance()
    progress.push()
    self.assertProgressSummary(1, 3, progress)