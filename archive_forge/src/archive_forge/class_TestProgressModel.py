import unittest
from subunit.progress_model import ProgressModel
class TestProgressModel(unittest.TestCase):

    def assertProgressSummary(self, pos, total, progress):
        """Assert that a progress model has reached a particular point."""
        self.assertEqual(pos, progress.pos())
        self.assertEqual(total, progress.width())

    def test_new_progress_0_0(self):
        progress = ProgressModel()
        self.assertProgressSummary(0, 0, progress)

    def test_advance_0_0(self):
        progress = ProgressModel()
        progress.advance()
        self.assertProgressSummary(1, 0, progress)

    def test_advance_1_0(self):
        progress = ProgressModel()
        progress.advance()
        self.assertProgressSummary(1, 0, progress)

    def test_set_width_absolute(self):
        progress = ProgressModel()
        progress.set_width(10)
        self.assertProgressSummary(0, 10, progress)

    def test_set_width_absolute_preserves_pos(self):
        progress = ProgressModel()
        progress.advance()
        progress.set_width(2)
        self.assertProgressSummary(1, 2, progress)

    def test_adjust_width(self):
        progress = ProgressModel()
        progress.adjust_width(10)
        self.assertProgressSummary(0, 10, progress)
        progress.adjust_width(-10)
        self.assertProgressSummary(0, 0, progress)

    def test_adjust_width_preserves_pos(self):
        progress = ProgressModel()
        progress.advance()
        progress.adjust_width(10)
        self.assertProgressSummary(1, 10, progress)
        progress.adjust_width(-10)
        self.assertProgressSummary(1, 0, progress)

    def test_push_preserves_progress(self):
        progress = ProgressModel()
        progress.adjust_width(3)
        progress.advance()
        progress.push()
        self.assertProgressSummary(1, 3, progress)

    def test_advance_advances_substack(self):
        progress = ProgressModel()
        progress.adjust_width(3)
        progress.advance()
        progress.push()
        progress.adjust_width(1)
        progress.advance()
        self.assertProgressSummary(2, 3, progress)

    def test_adjust_width_adjusts_substack(self):
        progress = ProgressModel()
        progress.adjust_width(3)
        progress.advance()
        progress.push()
        progress.adjust_width(2)
        progress.advance()
        self.assertProgressSummary(3, 6, progress)

    def test_set_width_adjusts_substack(self):
        progress = ProgressModel()
        progress.adjust_width(3)
        progress.advance()
        progress.push()
        progress.set_width(2)
        progress.advance()
        self.assertProgressSummary(3, 6, progress)

    def test_pop_restores_progress(self):
        progress = ProgressModel()
        progress.adjust_width(3)
        progress.advance()
        progress.push()
        progress.adjust_width(1)
        progress.advance()
        progress.pop()
        self.assertProgressSummary(1, 3, progress)