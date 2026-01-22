from typing import Dict, Any
import numpy as np
import ase
from ase.db import connect
from ase.calculators.calculator import Calculator
def _mangled_checkpoint_id(self):
    """
        Returns a mangled checkpoint id string:
            check_c_1:c_2:c_3:...
        E.g. if checkpoint is nested and id is [3,2,6] it returns:
            'check3:2:6'
        """
    return 'check' + ':'.join((str(id) for id in self.checkpoint_id))