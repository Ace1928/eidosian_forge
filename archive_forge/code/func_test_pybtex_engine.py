from __future__ import unicode_literals
import os
import posixpath
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp
import pytest
from pybtex import errors, io
from .utils import diff, get_data
@pytest.mark.parametrize(['filenames'], [(['cyrillic.bib', 'unsrt.bst'],), (['cyrillic.bib', 'plain.bst'],), (['cyrillic.bib', 'alpha.bst'],), (['extrafields.bib', 'unsrt.bst'],)])
@pytest.mark.parametrize(['check'], [(check_make_bibliography,), (check_format_from_string,)])
def test_pybtex_engine(check, filenames):
    import pybtex
    check(pybtex, filenames)