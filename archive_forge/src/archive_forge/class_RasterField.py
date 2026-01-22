from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
class RasterField(BaseSpatialField):
    """
    Raster field for GeoDjango -- evaluates into GDALRaster objects.
    """
    description = _('Raster Field')
    geom_type = 'RASTER'
    geography = False

    def _check_connection(self, connection):
        if not connection.features.gis_enabled or not connection.features.supports_raster:
            raise ImproperlyConfigured('Raster fields require backends with raster support.')

    def db_type(self, connection):
        self._check_connection(connection)
        return super().db_type(connection)

    def from_db_value(self, value, expression, connection):
        return connection.ops.parse_raster(value)

    def contribute_to_class(self, cls, name, **kwargs):
        super().contribute_to_class(cls, name, **kwargs)
        setattr(cls, self.attname, SpatialProxy(gdal.GDALRaster, self))

    def get_transform(self, name):
        from django.contrib.gis.db.models.lookups import RasterBandTransform
        try:
            band_index = int(name)
            return type('SpecificRasterBandTransform', (RasterBandTransform,), {'band_index': band_index})
        except ValueError:
            pass
        return super().get_transform(name)