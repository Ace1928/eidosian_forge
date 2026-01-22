import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
class DistanceOutputSpec(TraitedSpec):
    distance = traits.Float()
    point1 = traits.Array(shape=(3,))
    point2 = traits.Array(shape=(3,))
    histogram = File()