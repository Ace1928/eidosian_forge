import pytest
import subprocess
import json
import sys
from numpy.distutils import _shell_utils
from numpy.testing import IS_WASM

    Test that split is the inverse operation of join
    