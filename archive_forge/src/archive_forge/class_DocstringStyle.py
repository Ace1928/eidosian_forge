import enum
import typing as T
class DocstringStyle(enum.Enum):
    """Docstring style."""
    REST = 1
    GOOGLE = 2
    NUMPYDOC = 3
    EPYDOC = 4
    AUTO = 255