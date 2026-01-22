import os
import re as regex
from ..base import (
class SkullfinderOutputSpec(TraitedSpec):
    outputLabelFile = File(desc='path/name of label file')