import atexit
import json
import os
import shutil
import sys
import tempfile
from os import path as osp
from os.path import join as pjoin
from stat import S_IRGRP, S_IROTH, S_IRUSR
from tempfile import TemporaryDirectory
from unittest.mock import patch
import jupyter_core
import jupyterlab_server
from ipykernel.kernelspec import write_kernel_spec
from jupyter_server.serverapp import ServerApp
from jupyterlab_server.process_app import ProcessApp
from traitlets import default
class RootedServerApp(ServerApp):

    @default('root_dir')
    def _default_root_dir(self):
        """Create a temporary directory with some file structure."""
        root_dir = tempfile.mkdtemp(prefix='mock_root')
        os.mkdir(osp.join(root_dir, 'src'))
        with open(osp.join(root_dir, 'src', 'temp.txt'), 'w') as fid:
            fid.write('hello')
        readonly_filepath = osp.join(root_dir, 'src', 'readonly-temp.txt')
        with open(readonly_filepath, 'w') as fid:
            fid.write('hello from a readonly file')
        os.chmod(readonly_filepath, S_IRUSR | S_IRGRP | S_IROTH)
        atexit.register(lambda: shutil.rmtree(root_dir, True))
        return root_dir