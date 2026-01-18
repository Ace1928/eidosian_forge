import os
import shutil
import unittest
from monty.tempfile import ScratchDir
def test_with_copy_gzip(self):
    with open('pre_scratch_text', 'w') as f:
        f.write('write')
    init_gz = [f for f in os.listdir(os.getcwd()) if f.endswith('.gz')]
    with ScratchDir(self.scratch_root, copy_from_current_on_enter=True, copy_to_current_on_exit=True, gzip_on_exit=True):
        with open('scratch_text', 'w') as f:
            f.write('write')
    files = os.listdir(os.getcwd())
    assert 'scratch_text.gz' in files
    for f in files:
        if f.endswith('.gz') and f not in init_gz:
            os.remove(f)
    os.remove('pre_scratch_text')