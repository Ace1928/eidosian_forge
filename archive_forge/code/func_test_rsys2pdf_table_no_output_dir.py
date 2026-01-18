import shutil
import subprocess
import tempfile
import pytest
from chempy.util.table import rsys2tablines, rsys2table, rsys2pdf_table
from .test_graph import _get_rsys
from ..testing import skipif
@pytest.mark.skipif(pdflatex_missing, reason='latex not installed? (pdflatex command missing)')
def test_rsys2pdf_table_no_output_dir():
    rsys = _get_rsys()
    rsys2pdf_table(rsys, save=False)