from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
@property
def algorithms(self):
    """Returns the list of algorithms that are used to identify this coordination geometry."""
    return self._algorithms