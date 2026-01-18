import asyncio
import concurrent.futures
import copy
import datetime
import functools
import os
import re
import sys
import threading
import warnings
from base64 import b64decode, b64encode
from queue import Empty
from typing import Any
from unittest.mock import MagicMock, Mock
import nbformat
import pytest
import xmltodict
from flaky import flaky  # type:ignore
from jupyter_client import KernelClient, KernelManager
from jupyter_client._version import version_info
from jupyter_client.kernelspec import KernelSpecManager
from nbconvert.filters import strip_ansi
from nbformat import NotebookNode
from testpath import modified_env
from traitlets import TraitError
from nbclient import NotebookClient, execute
from nbclient.exceptions import CellExecutionError
from .base import NBClientTestsBase
def test_parallel_notebooks(capfd, tmpdir):
    """Two notebooks should be able to be run simultaneously without problems.

    The two notebooks spawned here use the filesystem to check that the other notebook
    wrote to the filesystem."""
    opts = {'kernel_name': 'python'}
    input_name = 'Parallel Execute {label}.ipynb'
    input_file = os.path.join(current_dir, 'files', input_name)
    res = notebook_resources()
    with modified_env({'NBEXECUTE_TEST_PARALLEL_TMPDIR': str(tmpdir)}):
        threads = [threading.Thread(target=run_notebook, args=(input_file.format(label=label), opts, res)) for label in ('A', 'B')]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)
    captured = capfd.readouterr()
    assert filter_messages_on_error_output(captured.err) == ''