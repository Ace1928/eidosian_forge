from django.apps import apps
from django.contrib.gis.db.models import GeometryField
from django.contrib.sitemaps import Sitemap
from django.db import models
from django.urls import reverse
class KMZSitemap(KMLSitemap):
    geo_format = 'kmz'