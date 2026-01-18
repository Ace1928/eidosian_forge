import os
import subprocess
import sys
import pytest
import matplotlib

    When using -OO or export PYTHONOPTIMIZE=2, docstrings are discarded,
    this simple test may prevent something like issue #17970.
    