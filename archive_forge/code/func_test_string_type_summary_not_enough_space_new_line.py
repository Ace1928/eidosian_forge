from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import custom_descriptions
from fire import testutils
def test_string_type_summary_not_enough_space_new_line(self):
    component = 'Test'
    summary = custom_descriptions.GetSummary(obj=component, available_space=4, line_length=LINE_LENGTH)
    self.assertEqual(summary, '"Test"')