from ..base import (
import os
class CleanerOutputSpec(TraitedSpec):
    cleaned_functional_file = File(exists=True, desc='Cleaned session data')