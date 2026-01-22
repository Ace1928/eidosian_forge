import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def _nameToTitle(self, name, forwardStringTitle=False):
    """
        Converts a function name to a title based on ``self.titleFormat``.

        Parameters
        ----------
        name: str
            Name of the function
        forwardStringTitle: bool
            If ``self.titleFormat`` is a string and ``forwardStrTitle`` is True,
            ``self.titleFormat`` will be used as the title. Otherwise, if
            ``self.titleFormat`` is *None*, the name will be returned unchanged.
            Finally, if ``self.titleFormat`` is a callable, it will be called with
            the name as an input and the output will be returned
        """
    titleFormat = self.titleFormat
    isString = isinstance(titleFormat, str)
    if titleFormat is None or (isString and (not forwardStringTitle)):
        return name
    elif isString:
        return titleFormat
    return titleFormat(name)