from copy import copy
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Callable, Dict, Union, List, Optional, no_type_check
import numpy as np
from scipy.special import erf
from pyquil.quilatom import TemplateWaveform, _update_envelope, _complex_str, Expression, substitute
def _optional_field_strs(wf: TemplateWaveform) -> List[str]:
    """Get the printed representations of optional template parameters."""
    result = []
    for field, spec in getattr(wf, '__dataclass_fields__', {}).items():
        if spec.default is None:
            value = getattr(wf, field, None)
            if value is not None:
                result.append(f'{field}: {value}')
    return result