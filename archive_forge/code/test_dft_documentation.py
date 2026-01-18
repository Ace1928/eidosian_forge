import os
import sqlite3
from io import BytesIO
from os.path import dirname
from os.path import join as pjoin
from ..testing import suppress_warnings
import unittest
import pytest
from .. import nifti1
from ..optpkg import optional_package
Build a dft database in memory to avoid cross-process races
    and not modify the host filesystem.