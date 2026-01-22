import os.path
from ._paml import Paml
from . import _parse_codeml
class CodemlError(EnvironmentError):
    """CODEML failed. Run with verbose=True to view CODEML's error message."""