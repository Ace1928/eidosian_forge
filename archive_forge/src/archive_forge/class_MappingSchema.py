from __future__ import annotations
import abc
import typing as t
from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth
from sqlglot.trie import TrieResult, in_trie, new_trie
class MappingSchema(AbstractMappingSchema, Schema):
    """
    Schema based on a nested mapping.

    Args:
        schema: Mapping in one of the following forms:
            1. {table: {col: type}}
            2. {db: {table: {col: type}}}
            3. {catalog: {db: {table: {col: type}}}}
            4. None - Tables will be added later
        visible: Optional mapping of which columns in the schema are visible. If not provided, all columns
            are assumed to be visible. The nesting should mirror that of the schema:
            1. {table: set(*cols)}}
            2. {db: {table: set(*cols)}}}
            3. {catalog: {db: {table: set(*cols)}}}}
        dialect: The dialect to be used for custom type mappings & parsing string arguments.
        normalize: Whether to normalize identifier names according to the given dialect or not.
    """

    def __init__(self, schema: t.Optional[t.Dict]=None, visible: t.Optional[t.Dict]=None, dialect: DialectType=None, normalize: bool=True) -> None:
        self.dialect = dialect
        self.visible = {} if visible is None else visible
        self.normalize = normalize
        self._type_mapping_cache: t.Dict[str, exp.DataType] = {}
        self._depth = 0
        schema = {} if schema is None else schema
        super().__init__(self._normalize(schema) if self.normalize else schema)

    @classmethod
    def from_mapping_schema(cls, mapping_schema: MappingSchema) -> MappingSchema:
        return MappingSchema(schema=mapping_schema.mapping, visible=mapping_schema.visible, dialect=mapping_schema.dialect, normalize=mapping_schema.normalize)

    def copy(self, **kwargs) -> MappingSchema:
        return MappingSchema(**{'schema': self.mapping.copy(), 'visible': self.visible.copy(), 'dialect': self.dialect, 'normalize': self.normalize, **kwargs})

    def add_table(self, table: exp.Table | str, column_mapping: t.Optional[ColumnMapping]=None, dialect: DialectType=None, normalize: t.Optional[bool]=None, match_depth: bool=True) -> None:
        """
        Register or update a table. Updates are only performed if a new column mapping is provided.
        The added table must have the necessary number of qualifiers in its path to match the schema's nesting level.

        Args:
            table: the `Table` expression instance or string representing the table.
            column_mapping: a column mapping that describes the structure of the table.
            dialect: the SQL dialect that will be used to parse `table` if it's a string.
            normalize: whether to normalize identifiers according to the dialect of interest.
            match_depth: whether to enforce that the table must match the schema's depth or not.
        """
        normalized_table = self._normalize_table(table, dialect=dialect, normalize=normalize)
        if match_depth and (not self.empty) and (len(normalized_table.parts) != self.depth()):
            raise SchemaError(f"Table {normalized_table.sql(dialect=self.dialect)} must match the schema's nesting level: {self.depth()}.")
        normalized_column_mapping = {self._normalize_name(key, dialect=dialect, normalize=normalize): value for key, value in ensure_column_mapping(column_mapping).items()}
        schema = self.find(normalized_table, raise_on_missing=False)
        if schema and (not normalized_column_mapping):
            return
        parts = self.table_parts(normalized_table)
        nested_set(self.mapping, tuple(reversed(parts)), normalized_column_mapping)
        new_trie([parts], self.mapping_trie)

    def column_names(self, table: exp.Table | str, only_visible: bool=False, dialect: DialectType=None, normalize: t.Optional[bool]=None) -> t.List[str]:
        normalized_table = self._normalize_table(table, dialect=dialect, normalize=normalize)
        schema = self.find(normalized_table)
        if schema is None:
            return []
        if not only_visible or not self.visible:
            return list(schema)
        visible = self.nested_get(self.table_parts(normalized_table), self.visible) or []
        return [col for col in schema if col in visible]

    def get_column_type(self, table: exp.Table | str, column: exp.Column | str, dialect: DialectType=None, normalize: t.Optional[bool]=None) -> exp.DataType:
        normalized_table = self._normalize_table(table, dialect=dialect, normalize=normalize)
        normalized_column_name = self._normalize_name(column if isinstance(column, str) else column.this, dialect=dialect, normalize=normalize)
        table_schema = self.find(normalized_table, raise_on_missing=False)
        if table_schema:
            column_type = table_schema.get(normalized_column_name)
            if isinstance(column_type, exp.DataType):
                return column_type
            elif isinstance(column_type, str):
                return self._to_data_type(column_type, dialect=dialect)
        return exp.DataType.build('unknown')

    def has_column(self, table: exp.Table | str, column: exp.Column | str, dialect: DialectType=None, normalize: t.Optional[bool]=None) -> bool:
        normalized_table = self._normalize_table(table, dialect=dialect, normalize=normalize)
        normalized_column_name = self._normalize_name(column if isinstance(column, str) else column.this, dialect=dialect, normalize=normalize)
        table_schema = self.find(normalized_table, raise_on_missing=False)
        return normalized_column_name in table_schema if table_schema else False

    def _normalize(self, schema: t.Dict) -> t.Dict:
        """
        Normalizes all identifiers in the schema.

        Args:
            schema: the schema to normalize.

        Returns:
            The normalized schema mapping.
        """
        normalized_mapping: t.Dict = {}
        flattened_schema = flatten_schema(schema, depth=dict_depth(schema) - 1)
        for keys in flattened_schema:
            columns = nested_get(schema, *zip(keys, keys))
            if not isinstance(columns, dict):
                raise SchemaError(f"Table {'.'.join(keys[:-1])} must match the schema's nesting level: {len(flattened_schema[0])}.")
            normalized_keys = [self._normalize_name(key, is_table=True) for key in keys]
            for column_name, column_type in columns.items():
                nested_set(normalized_mapping, normalized_keys + [self._normalize_name(column_name)], column_type)
        return normalized_mapping

    def _normalize_table(self, table: exp.Table | str, dialect: DialectType=None, normalize: t.Optional[bool]=None) -> exp.Table:
        dialect = dialect or self.dialect
        normalize = self.normalize if normalize is None else normalize
        normalized_table = exp.maybe_parse(table, into=exp.Table, dialect=dialect, copy=normalize)
        if normalize:
            for arg in exp.TABLE_PARTS:
                value = normalized_table.args.get(arg)
                if isinstance(value, exp.Identifier):
                    normalized_table.set(arg, normalize_name(value, dialect=dialect, is_table=True, normalize=normalize))
        return normalized_table

    def _normalize_name(self, name: str | exp.Identifier, dialect: DialectType=None, is_table: bool=False, normalize: t.Optional[bool]=None) -> str:
        return normalize_name(name, dialect=dialect or self.dialect, is_table=is_table, normalize=self.normalize if normalize is None else normalize).name

    def depth(self) -> int:
        if not self.empty and (not self._depth):
            self._depth = super().depth() - 1
        return self._depth

    def _to_data_type(self, schema_type: str, dialect: DialectType=None) -> exp.DataType:
        """
        Convert a type represented as a string to the corresponding `sqlglot.exp.DataType` object.

        Args:
            schema_type: the type we want to convert.
            dialect: the SQL dialect that will be used to parse `schema_type`, if needed.

        Returns:
            The resulting expression type.
        """
        if schema_type not in self._type_mapping_cache:
            dialect = dialect or self.dialect
            udt = Dialect.get_or_raise(dialect).SUPPORTS_USER_DEFINED_TYPES
            try:
                expression = exp.DataType.build(schema_type, dialect=dialect, udt=udt)
                self._type_mapping_cache[schema_type] = expression
            except AttributeError:
                in_dialect = f' in dialect {dialect}' if dialect else ''
                raise SchemaError(f"Failed to build type '{schema_type}'{in_dialect}.")
        return self._type_mapping_cache[schema_type]