from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.coordseq import GEOSCoordSeq
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.geometry import GEOSGeometry, LinearGeometryMixin
from django.contrib.gis.geos.point import Point
from django.contrib.gis.shortcuts import numpy
def _listarr(self, func):
    """
        Return a sequence (list) corresponding with the given function.
        Return a numpy array if possible.
        """
    lst = [func(i) for i in range(len(self))]
    if numpy:
        return numpy.array(lst)
    else:
        return lst