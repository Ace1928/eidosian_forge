from django.db.backends.postgresql.schema import DatabaseSchemaEditor
from django.db.models.expressions import Col, Func
def _create_index_sql(self, model, *, fields=None, **kwargs):
    if fields is None or len(fields) != 1 or (not hasattr(fields[0], 'geodetic')):
        return super()._create_index_sql(model, fields=fields, **kwargs)
    field = fields[0]
    expressions = None
    opclasses = None
    if field.geom_type == 'RASTER':
        expressions = Func(Col(None, field), template=self.rast_index_template)
        fields = None
    elif field.dim > 2 and (not field.geography):
        opclasses = [self.geom_index_ops_nd]
    name = kwargs.get('name')
    if not name:
        name = self._create_index_name(model._meta.db_table, [field.column], '_id')
    return super()._create_index_sql(model, fields=fields, name=name, using=' USING %s' % self.geom_index_type, opclasses=opclasses, expressions=expressions)