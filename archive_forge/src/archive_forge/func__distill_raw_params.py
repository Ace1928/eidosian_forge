from __future__ import annotations
import typing
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Tuple
from .. import exc
def _distill_raw_params(params: Optional[_DBAPIAnyExecuteParams]) -> _DBAPIMultiExecuteParams:
    if params is None:
        return _no_tuple
    elif isinstance(params, list):
        if params and (not isinstance(params[0], (tuple, Mapping))):
            raise exc.ArgumentError('List argument must consist only of tuples or dictionaries')
        return params
    elif isinstance(params, (tuple, dict)) or isinstance(params, Mapping):
        return [params]
    else:
        raise exc.ArgumentError('mapping or sequence expected for parameters')