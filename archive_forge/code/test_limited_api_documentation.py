import os
import shutil
import subprocess
import sys
import sysconfig
import pytest
from numpy.testing import IS_WASM
Test building a third-party C extension with the limited API.