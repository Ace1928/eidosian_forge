from django.contrib.gis.db.backends.base.models import SpatialRefSysMixin
from django.db import models
class SpatialiteSpatialRefSys(models.Model, SpatialRefSysMixin):
    """
    The 'spatial_ref_sys' table from SpatiaLite.
    """
    srid = models.IntegerField(primary_key=True)
    auth_name = models.CharField(max_length=256)
    auth_srid = models.IntegerField()
    ref_sys_name = models.CharField(max_length=256)
    proj4text = models.CharField(max_length=2048)
    srtext = models.CharField(max_length=2048)

    class Meta:
        app_label = 'gis'
        db_table = 'spatial_ref_sys'
        managed = False

    @property
    def wkt(self):
        return self.srtext