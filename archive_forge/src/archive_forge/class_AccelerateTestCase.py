import asyncio
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Union
from unittest import mock
import torch
import accelerate
from ..state import AcceleratorState, PartialState
from ..utils import (
class AccelerateTestCase(unittest.TestCase):
    """
    A TestCase class that will reset the accelerator state at the end of every test. Every test that checks or utilizes
    the `AcceleratorState` class should inherit from this to avoid silent failures due to state being shared between
    tests.
    """

    def tearDown(self):
        super().tearDown()
        AcceleratorState._reset_state()
        PartialState._reset_state()