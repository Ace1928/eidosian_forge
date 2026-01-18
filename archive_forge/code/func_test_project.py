import pytest
import os
import subprocess
import sys
import shutil
def test_project(self):
    try:
        subprocess.check_output([sys.executable or 'python', os.path.join(self.pinstall_path, 'main.py')], stderr=subprocess.STDOUT, env=self.env)
    except subprocess.CalledProcessError as e:
        print(e.output.decode('utf8'))
        raise