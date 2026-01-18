import abc
from collections.abc import Mapping
from typing import TypeVar, Generic
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
A generic visit method that will dispatch to the correct