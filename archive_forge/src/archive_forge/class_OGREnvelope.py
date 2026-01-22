from ctypes import Structure, c_double
from django.contrib.gis.gdal.error import GDALException
class OGREnvelope(Structure):
    """Represent the OGREnvelope C Structure."""
    _fields_ = [('MinX', c_double), ('MaxX', c_double), ('MinY', c_double), ('MaxY', c_double)]