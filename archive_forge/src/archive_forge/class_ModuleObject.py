from __future__ import annotations
import dataclasses
import typing as T
from .. import build, mesonlib
from ..build import IncludeDirs
from ..interpreterbase.decorators import noKwargs, noPosargs
from ..mesonlib import relpath, HoldableObject, MachineChoice
from ..programs import ExternalProgram
class ModuleObject(HoldableObject):
    """Base class for all objects returned by modules
    """

    def __init__(self) -> None:
        self.methods: T.Dict[str, T.Callable[[ModuleState, T.List['TYPE_var'], 'TYPE_kwargs'], T.Union[ModuleReturnValue, 'TYPE_var']]] = {}