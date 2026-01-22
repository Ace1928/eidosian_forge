from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
class ExtentField(Field):
    """Used as a return value from an extent aggregate"""
    description = _('Extent Aggregate Field')

    def get_internal_type(self):
        return 'ExtentField'

    def select_format(self, compiler, sql, params):
        select = compiler.connection.ops.select_extent
        return (select % sql if select else sql, params)