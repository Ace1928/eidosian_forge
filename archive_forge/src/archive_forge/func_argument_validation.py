import inspect
from functools import wraps
from typing import Any, Callable, Optional, Type, Union, get_type_hints
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from torch.utils.data.datapipes._typing import _DataPipeMeta
def argument_validation(f):
    signature = inspect.signature(f)
    hints = get_type_hints(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for argument_name, value in bound.arguments.items():
            if argument_name in hints and isinstance(hints[argument_name], _DataPipeMeta):
                hint = hints[argument_name]
                if not isinstance(value, IterDataPipe):
                    raise TypeError(f"Expected argument '{argument_name}' as a IterDataPipe, but found {type(value)}")
                if not value.type.issubtype(hint.type):
                    raise TypeError(f"Expected type of argument '{argument_name}' as a subtype of hint {hint.type}, but found {value.type}")
        return f(*args, **kwargs)
    return wrapper