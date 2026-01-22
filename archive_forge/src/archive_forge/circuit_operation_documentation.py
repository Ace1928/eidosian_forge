import math
from typing import (
import numpy as np
import sympy
from cirq import circuits, ops, protocols, value, study
from cirq._compat import cached_property, proper_repr
Returns a copy of this operation with an updated ParamResolver.

        Any existing parameter mappings will have their values updated given
        the provided mapping, and any new parameters will be added to the
        ParamResolver.

        Note that any resulting parameter mappings with no corresponding
        parameter in the base circuit will be omitted. These parameters do not
        apply to the `repetitions` field if that is parameterized.

        Args:
            param_values: A map or ParamResolver able to convert old param
                values to new param values. This map will be composed with any
                existing ParamResolver via single-step resolution.
            recursive: If True, resolves parameter values recursively over the
                resolver; otherwise performs a single resolution step. This
                behavior applies only to the passed-in mapping, for the current
                application. Existing parameters are never resolved recursively
                because a->b and b->a needs to be a valid mapping.

        Returns:
            A copy of this operation with its ParamResolver updated as specified
                by param_values.
        