import logging
from django.contrib.gis.db.models import GeometryField
from django.db import OperationalError
from django.db.backends.mysql.schema import DatabaseSchemaEditor
def _create_spatial_index_name(self, model, field):
    return '%s_%s_id' % (model._meta.db_table, field.column)