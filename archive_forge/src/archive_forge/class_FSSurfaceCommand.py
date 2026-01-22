import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
class FSSurfaceCommand(FSCommand):
    """Support for FreeSurfer surface-related functions.
    For some functions, if the output file is not specified starting with 'lh.'
    or 'rh.', FreeSurfer prepends the prefix from the input file to the output
    filename. Output out_file must be adjusted to accommodate this. By
    including the full path in the filename, we can also avoid this behavior.
    """

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