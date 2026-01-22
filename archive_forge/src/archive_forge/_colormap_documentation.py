from __future__ import annotations
import abc
from enum import Enum
from typing import TYPE_CHECKING
import numpy as np

        Return colors correspondsing to proportions in x

        Parameters
        ----------
        x :
            Values in the range [0, 1]. O maps to the start of the
            gradient, and 1 to the end of the gradient.
        