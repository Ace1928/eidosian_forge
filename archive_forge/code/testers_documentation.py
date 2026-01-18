import os
import re
import shutil
import sys
import tempfile
import zipfile
from glob import glob
from os.path import abspath
from os.path import join as pjoin
from subprocess import PIPE, Popen
import os
import sys
import {mod_name}
Create distutils distribution file

    Parameters
    ----------
    repo_path : str
        path to repository containing code and ``setup.py``
    out_dir : str
        path to which to write new distribution file
    setup_params: str
        parameters to pass to ``setup.py`` to create distribution.
    zipglob : str
        glob identifying expected output file.

    Returns
    -------
    out_fname : str
        filename of generated distribution file

    Examples
    --------
    Make, return a zipped sdist::

      make_dist('/path/to/repo', '/tmp/path', 'sdist --formats=zip', '*.zip')

    Make, return a binary egg::

      make_dist('/path/to/repo', '/tmp/path', 'bdist_egg', '*.egg')
    