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
def _install_kernel(self, kernel_name, kernel_spec):
    """Install a kernel spec to the data directory.

        Parameters
        ----------
        kernel_name: str
            Name of the kernel.
        kernel_spec: dict
            The kernel spec for the kernel
        """
    paths = jupyter_core.paths
    kernel_dir = pjoin(paths.jupyter_data_dir(), 'kernels', kernel_name)
    os.makedirs(kernel_dir)
    with open(pjoin(kernel_dir, 'kernel.json'), 'w') as f:
        f.write(json.dumps(kernel_spec))