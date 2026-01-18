import filecmp
import os
from pathlib import Path
import shutil
import sys
from matplotlib.testing import subprocess_run_for_testing
import pytest
def test_plot_html_show_source_link(tmp_path):
    parent = Path(__file__).parent
    shutil.copyfile(parent / 'tinypages/conf.py', tmp_path / 'conf.py')
    shutil.copytree(parent / 'tinypages/_static', tmp_path / '_static')
    doctree_dir = tmp_path / 'doctrees'
    (tmp_path / 'index.rst').write_text('\n.. plot::\n\n    plt.plot(range(2))\n')
    html_dir1 = tmp_path / '_build' / 'html1'
    build_sphinx_html(tmp_path, doctree_dir, html_dir1)
    assert len(list(html_dir1.glob('**/index-1.py'))) == 1
    html_dir2 = tmp_path / '_build' / 'html2'
    build_sphinx_html(tmp_path, doctree_dir, html_dir2, extra_args=['-D', 'plot_html_show_source_link=0'])
    assert len(list(html_dir2.glob('**/index-1.py'))) == 0