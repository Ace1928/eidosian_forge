import shutil
import subprocess
import tempfile
import pytest
from chempy.util.table import rsys2tablines, rsys2table, rsys2pdf_table
from .test_graph import _get_rsys
from ..testing import skipif
def test_rsys2tablines():
    assert rsys2tablines(_get_rsys(), tex=False) == ['1 & 2 A & -> & B & 3 & - & None']