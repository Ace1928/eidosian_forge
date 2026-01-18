import functools
import importlib
import os
import platform
import subprocess
import sys
import pytest
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper

    A decorator to run *func* in a subprocess and assert that it prints
    "success" *success_count* times and nothing on stderr.

    TkAgg tests seem to have interactions between tests, so isolate each test
    in a subprocess. See GH#18261
    