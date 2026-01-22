from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
def add_update_values(self, values):
    """
        Convert a dictionary of field name to value mappings into an update
        query. This is the entry point for the public update() method on
        querysets.
        """
    values_seq = []
    for name, val in values.items():
        field = self.get_meta().get_field(name)
        direct = not (field.auto_created and (not field.concrete)) or not field.concrete
        model = field.model._meta.concrete_model
        if not direct or (field.is_relation and field.many_to_many):
            raise FieldError('Cannot update model field %r (only non-relations and foreign keys permitted).' % field)
        if model is not self.get_meta().concrete_model:
            self.add_related_update(model, field, val)
            continue
        values_seq.append((field, model, val))
    return self.add_update_fields(values_seq)