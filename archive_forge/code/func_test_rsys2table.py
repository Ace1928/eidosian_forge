import shutil
import subprocess
import tempfile
import pytest
from chempy.util.table import rsys2tablines, rsys2table, rsys2pdf_table
from .test_graph import _get_rsys
from ..testing import skipif
def test_rsys2table():
    assert rsys2table(_get_rsys()) == '\n\\begin{table}\n\\centering\n\\label{tab:none}\n\\caption[None]{None}\n\\begin{tabular}{lllllll}\n\\toprule\nId. & Reactants &  & Products & {Rate constant} & Unit & Ref \\\\\n\\midrule\n1 & \\ensuremath{2 \\boldsymbol{A}} & \\ensuremath{\\rightarrow} &' + ' \\ensuremath{\\boldsymbol{B}} & \\ensuremath{3} & \\ensuremath{-} & None \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}'