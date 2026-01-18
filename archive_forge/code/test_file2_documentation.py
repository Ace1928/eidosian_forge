import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
 File objects close automatically when out of scope, but
        other objects remain open. 