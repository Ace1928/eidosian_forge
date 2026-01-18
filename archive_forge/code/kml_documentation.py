from django.apps import apps
from django.contrib.gis.db.models import GeometryField
from django.contrib.sitemaps import Sitemap
from django.db import models
from django.urls import reverse

        This method is overridden so the appropriate `geo_format` attribute
        is placed on each URL element.
        