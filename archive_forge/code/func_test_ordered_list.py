import unittest
import commonmark
def test_ordered_list(self):
    src_markdown = '\nThis is a ordered list:\n1. One\n2. Two\n3. Three\n'
    expected_rst = '\nThis is a ordered list:\n\n#. One\n#. Two\n#. Three\n'
    self.assertEqualRender(src_markdown, expected_rst)