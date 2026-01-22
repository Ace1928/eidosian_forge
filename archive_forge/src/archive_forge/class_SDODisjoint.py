import re
from django.contrib.gis.db import models
from django.contrib.gis.db.backends.base.operations import BaseSpatialOperations
from django.contrib.gis.db.backends.oracle.adapter import OracleSpatialAdapter
from django.contrib.gis.db.backends.utils import SpatialOperator
from django.contrib.gis.geos.geometry import GEOSGeometry, GEOSGeometryBase
from django.contrib.gis.geos.prototypes.io import wkb_r
from django.contrib.gis.measure import Distance
from django.db.backends.oracle.operations import DatabaseOperations
class SDODisjoint(SpatialOperator):
    sql_template = "SDO_GEOM.RELATE(%%(lhs)s, 'DISJOINT', %%(rhs)s, %s) = 'DISJOINT'" % DEFAULT_TOLERANCE