from copy import copy
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Callable, Dict, Union, List, Optional, no_type_check
import numpy as np
from scipy.special import erf
from pyquil.quilatom import TemplateWaveform, _update_envelope, _complex_str, Expression, substitute
@no_type_check
def _wf_from_dict(name: str, params: Dict[str, Union[Expression, Real, Complex]]) -> TemplateWaveform:
    """Construct a TemplateWaveform from a name and a dictionary of properties.

    :param name: The Quil-T name of the template.
    :param params: A mapping from parameter names to their corresponding values.
    :returns: A template waveform.
    """
    params = copy(params)
    if name not in _waveform_classes:
        raise ValueError(f'Unknown template waveform {name}.')
    cls = _waveform_classes[name]
    fields = getattr(cls, '__dataclass_fields__', {})
    for param, value in params.items():
        if param not in fields:
            raise ValueError(f"Unexpected parameter '{param}' in {name}.")
        if isinstance(value, Expression):
            value = substitute(value, {})
        if isinstance(value, Real):
            params[param] = float(value)
        elif isinstance(value, Complex):
            pass
        else:
            raise ValueError(f"Unable to resolve parameter '{param}' in template {name} to a constant value.")
    for field, spec in fields.items():
        if field not in params and spec.default is not None:
            raise ValueError(f"Missing parameter '{field}' in {name}.")
    return cls(**params)