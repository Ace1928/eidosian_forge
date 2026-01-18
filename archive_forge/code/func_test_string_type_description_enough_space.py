from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import custom_descriptions
from fire import testutils
def test_string_type_description_enough_space(self):
    component = 'Test'
    description = custom_descriptions.GetDescription(obj=component, available_space=80, line_length=LINE_LENGTH)
    self.assertEqual(description, 'The string "Test"')