import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class ClassicalAnd(LogicalBinaryOp):
    """
    WARNING: The operand order for ClassicalAnd has changed.  In pyQuil versions <= 1.9, AND had
    signature

        AND %source %target

    Now, AND has signature

        AND %target %source
    """
    op = 'AND'