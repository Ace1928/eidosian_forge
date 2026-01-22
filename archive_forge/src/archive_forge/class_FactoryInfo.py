import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
class FactoryInfo(typing.NamedTuple):
    """
    Structure that stores information about factory.

    Parameters
    ----------
    engine : str
        Name of underlying execution engine.
    partition : str
        Name of the partition format.
    experimental : bool
        Whether underlying engine is experimental-only.
    """
    engine: str
    partition: str
    experimental: bool