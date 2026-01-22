import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
class LabelGeometryOutputSpec(TraitedSpec):
    output_file = File(exists=True, desc='CSV file of geometry measures')