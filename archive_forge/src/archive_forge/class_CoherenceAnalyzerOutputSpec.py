import numpy as np
import tempfile
from ...utils.misc import package_check
from ...utils.filemanip import fname_presuffix
from .base import NitimeBaseInterface
from ..base import (
class CoherenceAnalyzerOutputSpec(TraitedSpec):
    coherence_array = traits.Array(desc='The pairwise coherence values between the ROIs')
    timedelay_array = traits.Array(desc='The pairwise time delays between the ROIs (in seconds)')
    coherence_csv = File(desc='A csv file containing the pairwise  coherence values')
    timedelay_csv = File(desc='A csv file containing the pairwise time delay values')
    coherence_fig = File(desc='Figure representing coherence values')
    timedelay_fig = File(desc='Figure representing coherence values')