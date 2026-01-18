import datetime
import warnings
from typing import Any, Literal, Optional, Sequence, Union
from langchain_core.utils import check_package_version
from typing_extensions import TypedDict
from langchain.chains.query_constructor.ir import (
def true(self) -> bool:
    return True