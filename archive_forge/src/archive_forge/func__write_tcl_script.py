import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
def _write_tcl_script(self):
    fid = open('snapshots.tcl', 'w')
    script = ['save_tiff $env(_SNAPSHOT_STEM)-lat.tif', 'make_lateral_view', 'rotate_brain_y 180', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-med.tif', 'make_lateral_view', 'rotate_brain_x 90', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-ven.tif', 'make_lateral_view', 'rotate_brain_x -90', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-dor.tif']
    if isdefined(self.inputs.six_images) and self.inputs.six_images:
        script.extend(['make_lateral_view', 'rotate_brain_y 90', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-pos.tif', 'make_lateral_view', 'rotate_brain_y -90', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-ant.tif'])
    script.append('exit')
    fid.write('\n'.join(script))
    fid.close()