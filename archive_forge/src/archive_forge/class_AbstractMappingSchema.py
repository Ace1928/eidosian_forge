from __future__ import annotations
import abc
import typing as t
from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth
from sqlglot.trie import TrieResult, in_trie, new_trie
class AbstractMappingSchema:

    def __init__(self, mapping: t.Optional[t.Dict]=None) -> None:
        self.mapping = mapping or {}
        self.mapping_trie = new_trie((tuple(reversed(t)) for t in flatten_schema(self.mapping, depth=self.depth())))
        self._supported_table_args: t.Tuple[str, ...] = tuple()

    @property
    def empty(self) -> bool:
        return not self.mapping

    def depth(self) -> int:
        return dict_depth(self.mapping)

    @property
    def supported_table_args(self) -> t.Tuple[str, ...]:
        if not self._supported_table_args and self.mapping:
            depth = self.depth()
            if not depth:
                self._supported_table_args = tuple()
            elif 1 <= depth <= 3:
                self._supported_table_args = exp.TABLE_PARTS[:depth]
            else:
                raise SchemaError(f'Invalid mapping shape. Depth: {depth}')
        return self._supported_table_args

    def table_parts(self, table: exp.Table) -> t.List[str]:
        if isinstance(table.this, exp.ReadCSV):
            return [table.this.name]
        return [table.text(part) for part in exp.TABLE_PARTS if table.text(part)]

    def find(self, table: exp.Table, raise_on_missing: bool=True) -> t.Optional[t.Any]:
        """
        Returns the schema of a given table.

        Args:
            table: the target table.
            raise_on_missing: whether to raise in case the schema is not found.

        Returns:
            The schema of the target table.
        """
        parts = self.table_parts(table)[0:len(self.supported_table_args)]
        value, trie = in_trie(self.mapping_trie, parts)
        if value == TrieResult.FAILED:
            return None
        if value == TrieResult.PREFIX:
            possibilities = flatten_schema(trie, depth=dict_depth(trie) - 1)
            if len(possibilities) == 1:
                parts.extend(possibilities[0])
            else:
                message = ', '.join(('.'.join(parts) for parts in possibilities))
                if raise_on_missing:
                    raise SchemaError(f'Ambiguous mapping for {table}: {message}.')
                return None
        return self.nested_get(parts, raise_on_missing=raise_on_missing)

    def nested_get(self, parts: t.Sequence[str], d: t.Optional[t.Dict]=None, raise_on_missing=True) -> t.Optional[t.Any]:
        return nested_get(d or self.mapping, *zip(self.supported_table_args, reversed(parts)), raise_on_missing=raise_on_missing)