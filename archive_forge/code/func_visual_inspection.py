import os
import traceback
import warnings
from os.path import join
from stat import ST_MTIME
import re
import runpy
from docutils import nodes
from docutils.parsers.rst.roles import set_classes
from subprocess import check_call, DEVNULL, CalledProcessError
from pathlib import Path
import matplotlib
def visual_inspection():
    """Manually inspect generated files."""
    import subprocess
    images = []
    text = []
    pdf = []
    for dir, pyname, outnames in creates():
        for outname in outnames:
            path = os.path.join(dir, outname)
            ext = path.rsplit('.', 1)[1]
            if ext == 'pdf':
                pdf.append(path)
            elif ext in ['csv', 'txt', 'out', 'css', 'LDA', 'rst']:
                text.append(path)
            else:
                images.append(path)
    subprocess.call(['eog'] + images)
    subprocess.call(['evince'] + pdf)
    subprocess.call(['more'] + text)