import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_path_content_summary(self):
    """path_content_summary should always talk about the canonical form."""
    source, txt_path, bin_path = self.create_cf_tree(txt_reader=_append_text, txt_writer=_remove_appended_text, dir='source')
    if not source.supports_content_filtering():
        return
    source.lock_read()
    self.addCleanup(source.unlock)
    expected_canonical_form = b'Foo Txt\nend string\n'
    with source.get_file(txt_path, filtered=True) as f:
        self.assertEqual(f.read(), expected_canonical_form)
    with source.get_file(txt_path, filtered=False) as f:
        self.assertEqual(f.read(), b'Foo Txt')
    result = source.path_content_summary('file1.txt')
    self.assertEqual(result, ('file', None, False, None))