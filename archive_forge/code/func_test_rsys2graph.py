import os
import subprocess
import shutil
import tempfile
from chempy import Reaction, ReactionSystem, Substance
from ..graph import rsys2dot, rsys2graph
from ..testing import requires, skipif
@requires('numpy')
@skipif(dot_missing, reason='graphviz not installed? (dot command missing)')
def test_rsys2graph():
    rsys = _get_rsys()
    tempdir = tempfile.mkdtemp()
    try:
        rsys2graph(rsys, os.path.join(tempdir, 'out.png'))
        rsys2graph(rsys, os.path.join(tempdir, 'out.ps'))
        try:
            subprocess.call(['dot2tex', '-v'])
        except Exception:
            pass
        else:
            rsys2graph(rsys, os.path.join(tempdir, 'out.tex'))
    finally:
        shutil.rmtree(tempdir)