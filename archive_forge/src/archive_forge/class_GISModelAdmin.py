from django.contrib.admin import ModelAdmin
from django.contrib.gis.db import models
from django.contrib.gis.forms import OSMWidget
class GISModelAdmin(GeoModelAdminMixin, ModelAdmin):
    pass