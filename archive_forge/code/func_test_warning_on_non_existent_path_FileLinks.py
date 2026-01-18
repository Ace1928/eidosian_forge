from tempfile import NamedTemporaryFile, mkdtemp
from os.path import split, join as pjoin, dirname
import pathlib
from unittest import TestCase, mock
import struct
import wave
from io import BytesIO
import pytest
from IPython.lib import display
from IPython.testing.decorators import skipif_not_numpy
def test_warning_on_non_existent_path_FileLinks():
    """FileLinks: Calling _repr_html_ on non-existent files returns a warning"""
    fls = display.FileLinks('example')
    assert fls._repr_html_().startswith('Path (<tt>example</tt>)')