import numbers
from typing import AbstractSet, Any, cast, TYPE_CHECKING, TypeVar
from typing_extensions import Self
import sympy
from typing_extensions import Protocol
from cirq import study
from cirq._doc import doc_private
class SupportsParameterization(Protocol):
    """An object that can be parameterized by Symbols and resolved
    via a ParamResolver"""

    @doc_private
    def _is_parameterized_(self) -> bool:
        """Whether the object is parameterized by any Symbols that require
        resolution. Returns True if the object has any unresolved Symbols
        and False otherwise."""

    @doc_private
    def _parameter_names_(self) -> AbstractSet[str]:
        """Returns a collection of string names of parameters that require
        resolution. If _is_parameterized_ is False, the collection is empty.
        The converse is not necessarily true, because some objects may report
        that they are parameterized when they contain symbolic constants which
        need to be evaluated, but no free symbols.
        """

    @doc_private
    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> Self:
        """Resolve the parameters in the effect."""