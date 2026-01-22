from ..base import (
import os
class TrainingOutputSpec(TraitedSpec):
    trained_wts_file = File(exists=True, desc='Trained-weights file')