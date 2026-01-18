import os
import shutil
import unittest
from monty.tempfile import ScratchDir
def test_with_copy_nodelete(self):
    with open('pre_scratch_text', 'w') as f:
        f.write('write')
    with ScratchDir(self.scratch_root, copy_from_current_on_enter=True, copy_to_current_on_exit=True, delete_removed_files=False) as d:
        with open('scratch_text', 'w') as f:
            f.write('write')
        files = os.listdir(d)
        assert 'scratch_text' in files
        assert 'empty_file.txt' in files
        assert 'pre_scratch_text' in files
        os.remove('pre_scratch_text')
    assert not os.path.exists(d)
    files = os.listdir('.')
    assert 'scratch_text' in files
    assert 'pre_scratch_text' in files
    os.remove('scratch_text')
    os.remove('pre_scratch_text')