import os
import numpy as np
import nibabel as nb
from ..interfaces.base import (
class SignalExtractionOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='tsv file containing the computed signals, with as many columns as there are labels and as many rows as there are timepoints in in_file, plus a header row with values from class_labels')