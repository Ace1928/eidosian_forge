import os
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import WBCommand
from ... import logging
class MetricResampleOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output metric')
    roi_file = File(desc='ROI of vertices that got data from valid source vertices')