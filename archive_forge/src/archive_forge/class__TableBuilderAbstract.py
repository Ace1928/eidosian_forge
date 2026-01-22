from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
class _TableBuilderAbstract(ABC):
    """
    Abstract builder for info table.
    """
    _lines: list[str]
    info: _BaseInfo

    @abstractmethod
    def get_lines(self) -> list[str]:
        """Product in a form of list of lines (strings)."""

    @property
    def data(self) -> DataFrame | Series:
        return self.info.data

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """Dtypes of each of the DataFrame's columns."""
        return self.info.dtypes

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""
        return self.info.dtype_counts

    @property
    def display_memory_usage(self) -> bool:
        """Whether to display memory usage."""
        return bool(self.info.memory_usage)

    @property
    def memory_usage_string(self) -> str:
        """Memory usage string with proper size qualifier."""
        return self.info.memory_usage_string

    @property
    def non_null_counts(self) -> Sequence[int]:
        return self.info.non_null_counts

    def add_object_type_line(self) -> None:
        """Add line with string representation of dataframe to the table."""
        self._lines.append(str(type(self.data)))

    def add_index_range_line(self) -> None:
        """Add line with range of indices to the table."""
        self._lines.append(self.data.index._summary())

    def add_dtypes_line(self) -> None:
        """Add summary line with dtypes present in dataframe."""
        collected_dtypes = [f'{key}({val:d})' for key, val in sorted(self.dtype_counts.items())]
        self._lines.append(f'dtypes: {', '.join(collected_dtypes)}')