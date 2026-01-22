import typing
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, List, Literal, Optional, Tuple, Union
from .params import Description, ParamArg, ParamVal
Gets the item with key. If key is ``ParamArg.DEFAULT``, return the
        item under the default parameter, or raise a ``ValueError`` if no
        default exists.