from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
class UnsupportedCompilationError(CompileError):
    """Raised when an operation is not supported by the given compiler.

    .. seealso::

        :ref:`faq_sql_expression_string`

        :ref:`error_l7de`
    """
    code = 'l7de'

    def __init__(self, compiler: Union[Compiled, TypeCompiler], element_type: Type[ClauseElement], message: Optional[str]=None):
        super().__init__("Compiler %r can't render element of type %s%s" % (compiler, element_type, ': %s' % message if message else ''))
        self.compiler = compiler
        self.element_type = element_type
        self.message = message

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (self.__class__, (self.compiler, self.element_type, self.message))