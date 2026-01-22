from django.db.models.lookups import (
class RelatedLookupMixin:

    def get_prep_lookup(self):
        if not isinstance(self.lhs, MultiColSource) and (not hasattr(self.rhs, 'resolve_expression')):
            self.rhs = get_normalized_value(self.rhs, self.lhs)[0]
            if self.prepare_rhs and hasattr(self.lhs.output_field, 'path_infos'):
                target_field = self.lhs.output_field.path_infos[-1].target_fields[-1]
                self.rhs = target_field.get_prep_value(self.rhs)
        return super().get_prep_lookup()

    def as_sql(self, compiler, connection):
        if isinstance(self.lhs, MultiColSource):
            assert self.rhs_is_direct_value()
            self.rhs = get_normalized_value(self.rhs, self.lhs)
            from django.db.models.sql.where import AND, WhereNode
            root_constraint = WhereNode()
            for target, source, val in zip(self.lhs.targets, self.lhs.sources, self.rhs):
                lookup_class = target.get_lookup(self.lookup_name)
                root_constraint.add(lookup_class(target.get_col(self.lhs.alias, source), val), AND)
            return root_constraint.as_sql(compiler, connection)
        return super().as_sql(compiler, connection)