from django.contrib.gis.db.backends.base.models import SpatialRefSysMixin
from django.db import models
class SpatialiteGeometryColumns(models.Model):
    """
    The 'geometry_columns' table from SpatiaLite.
    """
    f_table_name = models.CharField(max_length=256)
    f_geometry_column = models.CharField(max_length=256)
    coord_dimension = models.IntegerField()
    srid = models.IntegerField(primary_key=True)
    spatial_index_enabled = models.IntegerField()
    type = models.IntegerField(db_column='geometry_type')

    class Meta:
        app_label = 'gis'
        db_table = 'geometry_columns'
        managed = False

    def __str__(self):
        return '%s.%s - %dD %s field (SRID: %d)' % (self.f_table_name, self.f_geometry_column, self.coord_dimension, self.type, self.srid)

    @classmethod
    def table_name_col(cls):
        """
        Return the name of the metadata column used to store the feature table
        name.
        """
        return 'f_table_name'

    @classmethod
    def geom_col_name(cls):
        """
        Return the name of the metadata column used to store the feature
        geometry column.
        """
        return 'f_geometry_column'