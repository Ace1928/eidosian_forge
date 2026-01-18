import os
import subprocess
import sys
from textwrap import indent, dedent
import pytest
from numpy.testing import IS_WASM

    Ensures multiple "fake" static libraries are correctly linked.
    see gh-18295
    