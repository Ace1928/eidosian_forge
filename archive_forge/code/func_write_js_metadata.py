import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def write_js_metadata(pkg_data, project_shortname, has_wildcards):
    """Write an internal (not exported) R function to return all JS
    dependencies as required by dash.

    Parameters
    ----------
    project_shortname = hyphenated string, e.g. dash-html-components

    Returns
    -------
    """
    function_string = generate_js_metadata(pkg_data=pkg_data, project_shortname=project_shortname)
    file_name = 'internal.R'
    if not os.path.exists('R'):
        os.makedirs('R')
    file_path = os.path.join('R', file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(function_string)
        if has_wildcards:
            f.write(wildcard_helper)
    if os.path.exists('inst/deps'):
        shutil.rmtree('inst/deps')
    os.makedirs('inst/deps')
    for rel_dirname, _, filenames in os.walk(project_shortname):
        for filename in filenames:
            extension = os.path.splitext(filename)[1]
            if extension in ['.py', '.pyc', '.json']:
                continue
            target_dirname = os.path.join(os.path.join('inst/deps/', os.path.relpath(rel_dirname, project_shortname)))
            if not os.path.exists(target_dirname):
                os.makedirs(target_dirname)
            shutil.copy(os.path.join(rel_dirname, filename), target_dirname)