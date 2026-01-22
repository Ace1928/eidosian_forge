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
def _install_default_kernels(self):
    self._install_kernel(kernel_name='echo', kernel_spec={'argv': [sys.executable, '-m', 'jupyterlab.tests.echo_kernel', '-f', '{connection_file}'], 'display_name': 'Echo Kernel', 'language': 'echo'})
    paths = jupyter_core.paths
    ipykernel_dir = pjoin(paths.jupyter_data_dir(), 'kernels', 'ipython')
    write_kernel_spec(ipykernel_dir)