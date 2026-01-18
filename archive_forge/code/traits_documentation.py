import warnings
import numpy as np
from traitlets import TraitError, Undefined
from traittypes import Array, SciType
Example: shape_constraints(None,3) insists that the shape looks like (*,3)