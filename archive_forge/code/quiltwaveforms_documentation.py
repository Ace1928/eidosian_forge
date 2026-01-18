from copy import copy
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Callable, Dict, Union, List, Optional, no_type_check
import numpy as np
from scipy.special import erf
from pyquil.quilatom import TemplateWaveform, _update_envelope, _complex_str, Expression, substitute
 An optional frequency detuning factor. 