import inspect
import typing
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints
from adagio.instances import TaskContext
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_function, get_full_type_path
def _get_origin_type(anno: Any, assert_is_type: bool=True) -> Any:
    if anno is not None and anno.__module__ == 'typing':
        if anno is Any:
            return object
        if hasattr(typing, 'get_origin'):
            anno = typing.get_origin(anno)
        elif hasattr(anno, '__extra__'):
            anno = anno.__extra__
        elif hasattr(anno, '__origin__'):
            anno = anno.__origin__
    if anno is None:
        anno = type(None)
    if assert_is_type:
        assert_or_throw(_is_native_type(anno), TypeError(f"Can't find python type for {anno}"))
    return anno