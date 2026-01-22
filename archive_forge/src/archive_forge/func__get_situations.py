import os
from typing import Any, List
import numpy as np
from parlai.core.teachers import FixedDialogTeacher
from .build import build
def _get_situations(self):
    new_data = []
    for ep in self.data:
        new_data.append(ep[0])
    self.data = new_data