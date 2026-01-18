import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import NoReturn
from typing import Optional
from typing import Protocol
from typing import Type
from typing import TypeVar
Import and return the requested module ``modname``, or skip the
    current test if the module cannot be imported.

    :param modname:
        The name of the module to import.
    :param minversion:
        If given, the imported module's ``__version__`` attribute must be at
        least this minimal version, otherwise the test is still skipped.
    :param reason:
        If given, this reason is shown as the message when the module cannot
        be imported.

    :returns:
        The imported module. This should be assigned to its canonical name.

    Example::

        docutils = pytest.importorskip("docutils")
    