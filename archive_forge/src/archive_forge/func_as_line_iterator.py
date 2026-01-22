from io import IOBase, TextIOWrapper
from typing import Iterable, Any, Union, Type, Optional
def as_line_iterator(self, newline: Optional[str]='', encoding: Optional[str]='utf-8', **kwargs) -> Iterable[str]:
    """
        Return an iterator the yields lines from the stream.
        """
    return TextIOWrapper(self, newline=newline, encoding=encoding, **kwargs)