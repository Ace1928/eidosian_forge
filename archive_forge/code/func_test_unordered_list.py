import unittest
import commonmark
def test_unordered_list(self):
    src_markdown = '\nThis is a list:\n* List item\n* List item\n* List item\n'
    expected_rst = '\nThis is a list:\n\n* List item\n* List item\n* List item\n'
    self.assertEqualRender(src_markdown, expected_rst)