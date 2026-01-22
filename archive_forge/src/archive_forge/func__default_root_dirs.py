import os
import json
from jupyter_core.paths import jupyter_path
import nbconvert.exporters.templateexporter
def _default_root_dirs():
    root_dirs = []
    if DEV_MODE:
        root_dirs.append(os.path.abspath(os.path.join(ROOT, '..', 'share', 'jupyter')))
    if nbconvert.exporters.templateexporter.DEV_MODE:
        root_dirs.append(os.path.abspath(os.path.join(nbconvert.exporters.templateexporter.ROOT, '..', '..', 'share', 'jupyter')))
    root_dirs.extend(jupyter_path())
    return root_dirs