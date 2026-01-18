import collections
from osc_lib.cli import format_columns
from osc_lib.tests import utils
def test_size_column(self):
    content = 1576395005
    col = format_columns.SizeColumn(content)
    self.assertEqual(content, col.machine_readable())
    self.assertEqual('1.6G', col.human_readable())