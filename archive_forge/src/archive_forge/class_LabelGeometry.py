import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class LabelGeometry(ANTSCommand):
    """
    Extracts geometry measures using a label file and an optional image file

    Examples
    --------
    >>> from nipype.interfaces.ants import LabelGeometry
    >>> label_extract = LabelGeometry()
    >>> label_extract.inputs.dimension = 3
    >>> label_extract.inputs.label_image = 'atlas.nii.gz'
    >>> label_extract.cmdline
    'LabelGeometryMeasures 3 atlas.nii.gz [] atlas.csv'

    >>> label_extract.inputs.intensity_image = 'ants_Warp.nii.gz'
    >>> label_extract.cmdline
    'LabelGeometryMeasures 3 atlas.nii.gz ants_Warp.nii.gz atlas.csv'

    """
    _cmd = 'LabelGeometryMeasures'
    input_spec = LabelGeometryInputSpec
    output_spec = LabelGeometryOutputSpec