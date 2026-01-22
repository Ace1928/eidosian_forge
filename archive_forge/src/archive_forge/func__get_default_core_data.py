import json
import os.path as osp
from itertools import filterfalse
from .jlpmapp import HERE
def _get_default_core_data():
    """Get the data for the app template."""
    with open(pjoin(HERE, 'staging', 'package.json')) as fid:
        return json.load(fid)