import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class Unifize(AFNICommand):
    """3dUnifize - for uniformizing image intensity

    * The input dataset is supposed to be a T1-weighted volume,
      possibly already skull-stripped (e.g., via 3dSkullStrip).
      However, this program can be a useful step to take BEFORE
      3dSkullStrip, since the latter program can fail if the input
      volume is strongly shaded -- 3dUnifize will (mostly) remove
      such shading artifacts.

    * The output dataset has the white matter (WM) intensity approximately
      uniformized across space, and scaled to peak at about 1000.

    * The output dataset is always stored in float format!

    * If the input dataset has more than 1 sub-brick, only sub-brick
      #0 will be processed!

    * Want to correct EPI datasets for nonuniformity?
      You can try the new and experimental [Mar 2017] '-EPI' option.

    * The principal motive for this program is for use in an image
      registration script, and it may or may not be useful otherwise.

    * This program replaces the older (and very different) 3dUniformize,
      which is no longer maintained and may sublimate at any moment.
      (In other words, we do not recommend the use of 3dUniformize.)

    For complete details, see the `3dUnifize Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dUnifize.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> unifize = afni.Unifize()
    >>> unifize.inputs.in_file = 'structural.nii'
    >>> unifize.inputs.out_file = 'structural_unifized.nii'
    >>> unifize.cmdline
    '3dUnifize -prefix structural_unifized.nii -input structural.nii'
    >>> res = unifize.run()  # doctest: +SKIP

    """
    _cmd = '3dUnifize'
    input_spec = UnifizeInputSpec
    output_spec = UnifizeOutputSpec