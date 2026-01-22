import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
class MemoryMappedTable(TableBlock):
    """
    The table is said memory mapped when it doesn't use the user's RAM but loads the data
    from the disk instead.

    Pickling it doesn't copy the data into memory.
    Instead, only the path to the memory mapped arrow file is pickled, as well as the list
    of transforms to "replay" when reloading the table from the disk.

    Its implementation requires to store an history of all the transforms that were applied
    to the underlying pyarrow Table, so that they can be "replayed" when reloading the Table
    from the disk.

    This is different from the `InMemoryTable` table, for which pickling does copy all the
    data in memory.

    `InMemoryTable` must be used when data fit in memory, while `MemoryMapped` are reserved for
    data bigger than memory or when you want the memory footprint of your application to
    stay low.
    """

    def __init__(self, table: pa.Table, path: str, replays: Optional[List[Replay]]=None):
        super().__init__(table)
        self.path = os.path.abspath(path)
        self.replays: List[Replay] = replays if replays is not None else []

    @classmethod
    def from_file(cls, filename: str, replays=None):
        table = _memory_mapped_arrow_table_from_file(filename)
        table = cls._apply_replays(table, replays)
        return cls(table, filename, replays)

    def __getstate__(self):
        return {'path': self.path, 'replays': self.replays}

    def __setstate__(self, state):
        path = state['path']
        replays = state['replays']
        table = _memory_mapped_arrow_table_from_file(path)
        table = self._apply_replays(table, replays)
        MemoryMappedTable.__init__(self, table, path=path, replays=replays)

    @staticmethod
    def _apply_replays(table: pa.Table, replays: Optional[List[Replay]]=None) -> pa.Table:
        if replays is not None:
            for name, args, kwargs in replays:
                if name == 'cast':
                    table = table_cast(table, *args, **kwargs)
                elif name == 'flatten':
                    table = table_flatten(table, *args, **kwargs)
                else:
                    table = getattr(table, name)(*args, **kwargs)
        return table

    def _append_replay(self, replay: Replay) -> List[Replay]:
        replays = copy.deepcopy(self.replays)
        replays.append(replay)
        return replays

    def slice(self, offset=0, length=None):
        """
        Compute zero-copy slice of this Table.

        Args:
            offset (`int`, defaults to `0`):
                Offset from start of table to slice.
            length (`int`, defaults to `None`):
                Length of slice (default is until end of table starting from
                offset).

        Returns:
            `datasets.table.Table`
        """
        replay = ('slice', (offset, length), {})
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.fast_slice(offset=offset, length=length), self.path, replays)

    def filter(self, *args, **kwargs):
        """
        Select records from a Table. See `pyarrow.compute.filter` for full usage.
        """
        replay = ('filter', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.filter(*args, **kwargs), self.path, replays)

    def flatten(self, *args, **kwargs):
        """
        Flatten this Table.  Each column with a struct type is flattened
        into one column per struct field.  Other columns are left unchanged.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        replay = ('flatten', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_flatten(self.table, *args, **kwargs), self.path, replays)

    def combine_chunks(self, *args, **kwargs):
        """
        Make a new table by combining the chunks this table has.

        All the underlying chunks in the ChunkedArray of each column are
        concatenated into zero or one chunk.

        Args:
            memory_pool (`MemoryPool`, defaults to `None`):
                For memory allocations, if required, otherwise use default pool.

        Returns:
            `datasets.table.Table`
        """
        replay = ('combine_chunks', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.combine_chunks(*args, **kwargs), self.path, replays)

    def cast(self, *args, **kwargs):
        """
        Cast table values to another schema

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """
        replay = ('cast', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(table_cast(self.table, *args, **kwargs), self.path, replays)

    def replace_schema_metadata(self, *args, **kwargs):
        """
        EXPERIMENTAL: Create shallow copy of table by replacing schema
        key-value metadata with the indicated new metadata (which may be None,
        which deletes any existing metadata.

        Args:
            metadata (`dict`, defaults to `None`):

        Returns:
            `datasets.table.Table`: shallow_copy
        """
        replay = ('replace_schema_metadata', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.replace_schema_metadata(*args, **kwargs), self.path, replays)

    def add_column(self, *args, **kwargs):
        """
        Add column to Table at position.

        A new table is returned with the column added, the original table
        object is left unchanged.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`: New table with the passed column added.
        """
        replay = ('add_column', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.add_column(*args, **kwargs), self.path, replays)

    def append_column(self, *args, **kwargs):
        """
        Append column at end of columns.

        Args:
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column added.
        """
        replay = ('append_column', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.append_column(*args, **kwargs), self.path, replays)

    def remove_column(self, *args, **kwargs):
        """
        Create new Table with the indicated column removed.

        Args:
            i (`int`):
                Index of column to remove.

        Returns:
            `datasets.table.Table`:
                New table without the column.
        """
        replay = ('remove_column', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.remove_column(*args, **kwargs), self.path, replays)

    def set_column(self, *args, **kwargs):
        """
        Replace column in Table at position.

        Args:
            i (`int`):
                Index to place the column at.
            field_ (`Union[str, pyarrow.Field]`):
                If a string is passed then the type is deduced from the column
                data.
            column (`Union[pyarrow.Array, List[pyarrow.Array]]`):
                Column data.

        Returns:
            `datasets.table.Table`:
                New table with the passed column set.
        """
        replay = ('set_column', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.set_column(*args, **kwargs), self.path, replays)

    def rename_columns(self, *args, **kwargs):
        """
        Create new table with columns renamed to provided names.
        """
        replay = ('rename_columns', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.rename_columns(*args, **kwargs), self.path, replays)

    def drop(self, *args, **kwargs):
        """
        Drop one or more columns and return a new table.

        Args:
            columns (`List[str]`):
                List of field names referencing existing columns.

        Raises:
            `KeyError` : if any of the passed columns name are not existing.

        Returns:
            `datasets.table.Table`:
                New table without the columns.
        """
        replay = ('drop', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.drop(*args, **kwargs), self.path, replays)

    def select(self, *args, **kwargs):
        """
        Select columns of the table.

        Returns a new table with the specified columns, and metadata preserved.

        Args:
            columns (:obj:`Union[List[str], List[int]]`):
                The column names or integer indices to select.

        Returns:
            :class:`datasets.table.Table`: New table with the specified columns, and metadata preserved.
        """
        replay = ('select', copy.deepcopy(args), copy.deepcopy(kwargs))
        replays = self._append_replay(replay)
        return MemoryMappedTable(self.table.select(*args, **kwargs), self.path, replays)