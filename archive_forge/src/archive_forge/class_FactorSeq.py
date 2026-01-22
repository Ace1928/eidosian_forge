from __future__ import annotations
import logging # isort:skip
import typing as tp
from .bases import Init, SingleParameterizedProperty
from .container import Seq, Tuple
from .either import Either
from .primitive import String
from .singletons import Intrinsic
class FactorSeq(SingleParameterizedProperty[FactorSeqType]):
    """ Represents a collection of categorical factors. """

    def __init__(self, default: Init[FactorSeqType]=Intrinsic, *, help: str | None=None) -> None:
        type_param = Either(Seq(L1Factor), Seq(L2Factor), Seq(L3Factor))
        super().__init__(type_param, default=default, help=help)