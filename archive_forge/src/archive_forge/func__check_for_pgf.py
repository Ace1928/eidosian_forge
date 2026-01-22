from pathlib import Path
from tempfile import TemporaryDirectory
import locale
import logging
import os
import subprocess
import sys
import matplotlib as mpl
from matplotlib import _api
def _check_for_pgf(texsystem):
    """
    Check if a given TeX system + pgf is available

    Parameters
    ----------
    texsystem : str
        The executable name to check
    """
    with TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir, 'test.tex')
        tex_path.write_text('\n            \\documentclass{article}\n            \\usepackage{pgf}\n            \\begin{document}\n            \\typeout{pgfversion=\\pgfversion}\n            \\makeatletter\n            \\@@end\n        ', encoding='utf-8')
        try:
            subprocess.check_call([texsystem, '-halt-on-error', str(tex_path)], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return False
        return True