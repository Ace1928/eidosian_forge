import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
@staticmethod
def _associated_file(in_file, out_name):
    """Based on MRIsBuildFileName in freesurfer/utils/mrisurf.c

        If no path information is provided for out_name, use path and
        hemisphere (if also unspecified) from in_file to determine the path
        of the associated file.
        Use in_file prefix to indicate hemisphere for out_name, rather than
        inspecting the surface data structure.
        """
    path, base = os.path.split(out_name)
    if path == '':
        path, in_file = os.path.split(in_file)
        hemis = ('lh.', 'rh.')
        if in_file[:3] in hemis and base[:3] not in hemis:
            base = in_file[:3] + base
    return os.path.join(path, base)