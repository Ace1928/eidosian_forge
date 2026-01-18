import filecmp
import os
from pathlib import Path
import shutil
import sys
from matplotlib.testing import subprocess_run_for_testing
import pytest
@pytest.mark.parametrize('plot_html_show_source_link', [0, 1])
def test_show_source_link_false(tmp_path, plot_html_show_source_link):
    parent = Path(__file__).parent
    shutil.copyfile(parent / 'tinypages/conf.py', tmp_path / 'conf.py')
    shutil.copytree(parent / 'tinypages/_static', tmp_path / '_static')
    doctree_dir = tmp_path / 'doctrees'
    (tmp_path / 'index.rst').write_text('\n.. plot::\n    :show-source-link: false\n\n    plt.plot(range(2))\n')
    html_dir = tmp_path / '_build' / 'html'
    build_sphinx_html(tmp_path, doctree_dir, html_dir, extra_args=['-D', f'plot_html_show_source_link={plot_html_show_source_link}'])
    assert len(list(html_dir.glob('**/index-1.py'))) == 0