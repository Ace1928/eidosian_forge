from typing import Any, Iterable
from .version import __version__ as internal_version
from ._vendor.packaging.version import Version, InvalidVersion
def _convert_to_version(self, inp: Any) -> Any:
    if isinstance(inp, Version):
        return inp
    elif isinstance(inp, str):
        return Version(inp)
    elif isinstance(inp, Iterable):
        return Version('.'.join((str(item) for item in inp)))
    else:
        raise InvalidVersion(inp)