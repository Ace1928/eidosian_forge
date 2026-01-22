import warnings
from django.core.exceptions import FullResultSet
from django.db.models.sql.constants import INNER, LOUTER
from django.utils.deprecation import RemovedInDjango60Warning
class Join:
    """
    Used by sql.Query and sql.SQLCompiler to generate JOIN clauses into the
    FROM entry. For example, the SQL generated could be
        LEFT OUTER JOIN "sometable" T1
        ON ("othertable"."sometable_id" = "sometable"."id")

    This class is primarily used in Query.alias_map. All entries in alias_map
    must be Join compatible by providing the following attributes and methods:
        - table_name (string)
        - table_alias (possible alias for the table, can be None)
        - join_type (can be None for those entries that aren't joined from
          anything)
        - parent_alias (which table is this join's parent, can be None similarly
          to join_type)
        - as_sql()
        - relabeled_clone()
    """

    def __init__(self, table_name, parent_alias, table_alias, join_type, join_field, nullable, filtered_relation=None):
        self.table_name = table_name
        self.parent_alias = parent_alias
        self.table_alias = table_alias
        self.join_type = join_type
        if hasattr(join_field, 'get_joining_fields'):
            self.join_fields = join_field.get_joining_fields()
            self.join_cols = tuple(((lhs_field.column, rhs_field.column) for lhs_field, rhs_field in self.join_fields))
        else:
            warnings.warn('The usage of get_joining_columns() in Join is deprecated. Implement get_joining_fields() instead.', RemovedInDjango60Warning)
            self.join_fields = None
            self.join_cols = join_field.get_joining_columns()
        self.join_field = join_field
        self.nullable = nullable
        self.filtered_relation = filtered_relation

    def as_sql(self, compiler, connection):
        """
        Generate the full
           LEFT OUTER JOIN sometable ON sometable.somecol = othertable.othercol, params
        clause for this join.
        """
        join_conditions = []
        params = []
        qn = compiler.quote_name_unless_alias
        qn2 = connection.ops.quote_name
        join_fields = self.join_fields or self.join_cols
        for lhs, rhs in join_fields:
            if isinstance(lhs, str):
                lhs_full_name = '%s.%s' % (qn(self.parent_alias), qn2(lhs))
                rhs_full_name = '%s.%s' % (qn(self.table_alias), qn2(rhs))
            else:
                lhs, rhs = connection.ops.prepare_join_on_clause(self.parent_alias, lhs, self.table_alias, rhs)
                lhs_sql, lhs_params = compiler.compile(lhs)
                lhs_full_name = lhs_sql % lhs_params
                rhs_sql, rhs_params = compiler.compile(rhs)
                rhs_full_name = rhs_sql % rhs_params
            join_conditions.append(f'{lhs_full_name} = {rhs_full_name}')
        extra_cond = self.join_field.get_extra_restriction(self.table_alias, self.parent_alias)
        if extra_cond:
            extra_sql, extra_params = compiler.compile(extra_cond)
            join_conditions.append('(%s)' % extra_sql)
            params.extend(extra_params)
        if self.filtered_relation:
            try:
                extra_sql, extra_params = compiler.compile(self.filtered_relation)
            except FullResultSet:
                pass
            else:
                join_conditions.append('(%s)' % extra_sql)
                params.extend(extra_params)
        if not join_conditions:
            declared_field = getattr(self.join_field, 'field', self.join_field)
            raise ValueError('Join generated an empty ON clause. %s did not yield either joining columns or extra restrictions.' % declared_field.__class__)
        on_clause_sql = ' AND '.join(join_conditions)
        alias_str = '' if self.table_alias == self.table_name else ' %s' % self.table_alias
        sql = '%s %s%s ON (%s)' % (self.join_type, qn(self.table_name), alias_str, on_clause_sql)
        return (sql, params)

    def relabeled_clone(self, change_map):
        new_parent_alias = change_map.get(self.parent_alias, self.parent_alias)
        new_table_alias = change_map.get(self.table_alias, self.table_alias)
        if self.filtered_relation is not None:
            filtered_relation = self.filtered_relation.relabeled_clone(change_map)
        else:
            filtered_relation = None
        return self.__class__(self.table_name, new_parent_alias, new_table_alias, self.join_type, self.join_field, self.nullable, filtered_relation=filtered_relation)

    @property
    def identity(self):
        return (self.__class__, self.table_name, self.parent_alias, self.join_field, self.filtered_relation)

    def __eq__(self, other):
        if not isinstance(other, Join):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self):
        return hash(self.identity)

    def demote(self):
        new = self.relabeled_clone({})
        new.join_type = INNER
        return new

    def promote(self):
        new = self.relabeled_clone({})
        new.join_type = LOUTER
        return new