import unittest
import commonmark
def test_paragraph(self):
    src_markdown = 'Hello paragraph'
    expected_rst = '\nHello paragraph\n'
    self.assertEqualRender(src_markdown, expected_rst)