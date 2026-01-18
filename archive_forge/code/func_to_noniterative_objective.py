import copy
from typing import Any, Callable, Dict, Optional, Tuple, no_type_check
from fugue._utils.interfaceless import is_class_method
from triad import assert_or_throw
from triad.collections.function_wrapper import (
from triad.utils.convert import get_caller_global_local_vars, to_function
from tune.concepts.flow import Trial, TrialReport
from tune.exceptions import TuneCompileError
from tune.noniterative.objective import NonIterativeObjectiveFunc
def to_noniterative_objective(obj: Any, min_better: bool=True, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None) -> NonIterativeObjectiveFunc:
    if isinstance(obj, NonIterativeObjectiveFunc):
        return copy.copy(obj)
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    try:
        f = to_function(obj, global_vars=global_vars, local_vars=local_vars)
        if isinstance(f, NonIterativeObjectiveFunc):
            return copy.copy(f)
        return _NonIterativeObjectiveFuncWrapper.from_func(f, min_better)
    except Exception as e:
        exp = e
    raise TuneCompileError(f'{obj} is not a valid tunable function', exp)