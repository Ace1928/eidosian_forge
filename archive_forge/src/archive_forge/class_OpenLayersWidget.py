import logging
import warnings
from django.conf import settings
from django.contrib.gis import gdal
from django.contrib.gis.geometry import json_regex
from django.contrib.gis.geos import GEOSException, GEOSGeometry
from django.forms.widgets import Widget
from django.utils import translation
from django.utils.deprecation import RemovedInDjango51Warning
class OpenLayersWidget(BaseGeometryWidget):
    template_name = 'gis/openlayers.html'
    map_srid = 3857

    class Media:
        css = {'all': ('https://cdn.jsdelivr.net/npm/ol@v7.2.2/ol.css', 'gis/css/ol3.css')}
        js = ('https://cdn.jsdelivr.net/npm/ol@v7.2.2/dist/ol.js', 'gis/js/OLMapWidget.js')

    def serialize(self, value):
        return value.json if value else ''

    def deserialize(self, value):
        geom = super().deserialize(value)
        if geom and json_regex.match(value) and (self.map_srid != 4326):
            geom.srid = self.map_srid
        return geom