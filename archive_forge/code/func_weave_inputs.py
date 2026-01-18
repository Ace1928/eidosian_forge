import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
def weave_inputs(spec):
    return spec['config']['panelConfig']['exp']['fromOp']['inputs']