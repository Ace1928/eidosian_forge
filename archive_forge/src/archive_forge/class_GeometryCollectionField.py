from collections import defaultdict, namedtuple
from django.contrib.gis import forms, gdal
from django.contrib.gis.db.models.proxy import SpatialProxy
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.geos import (
from django.core.exceptions import ImproperlyConfigured
from django.db.models import Field
from django.utils.translation import gettext_lazy as _
class GeometryCollectionField(GeometryField):
    geom_type = 'GEOMETRYCOLLECTION'
    geom_class = GeometryCollection
    form_class = forms.GeometryCollectionField
    description = _('Geometry collection')