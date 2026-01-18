import pytest
import os
import subprocess
import sys
import shutil
def test_packaged_project(self):
    try:
        subprocess.check_output(os.path.join(self.pinstall_path, 'dist', 'main', 'main'), stderr=subprocess.STDOUT, env=self.get_run_env())
    except subprocess.CalledProcessError as e:
        print(e.output.decode('utf8'))
        raise