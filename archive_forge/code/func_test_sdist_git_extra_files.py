import glob
import os
import sys
import tarfile
import fixtures
from pbr.tests import base
def test_sdist_git_extra_files(self):
    """Test that extra files found in git are correctly added."""
    tf_path = glob.glob(os.path.join('dist', '*.tar.gz'))[0]
    tf = tarfile.open(tf_path)
    names = ['/'.join(p.split('/')[1:]) for p in tf.getnames()]
    self.assertIn('git-extra-file.txt', names)