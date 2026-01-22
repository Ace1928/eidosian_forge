from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def GEO_KEYS():
    return {1024: 'GTModelTypeGeoKey', 1025: 'GTRasterTypeGeoKey', 1026: 'GTCitationGeoKey', 2048: 'GeographicTypeGeoKey', 2049: 'GeogCitationGeoKey', 2050: 'GeogGeodeticDatumGeoKey', 2051: 'GeogPrimeMeridianGeoKey', 2052: 'GeogLinearUnitsGeoKey', 2053: 'GeogLinearUnitSizeGeoKey', 2054: 'GeogAngularUnitsGeoKey', 2055: 'GeogAngularUnitsSizeGeoKey', 2056: 'GeogEllipsoidGeoKey', 2057: 'GeogSemiMajorAxisGeoKey', 2058: 'GeogSemiMinorAxisGeoKey', 2059: 'GeogInvFlatteningGeoKey', 2060: 'GeogAzimuthUnitsGeoKey', 2061: 'GeogPrimeMeridianLongGeoKey', 2062: 'GeogTOWGS84GeoKey', 3059: 'ProjLinearUnitsInterpCorrectGeoKey', 3072: 'ProjectedCSTypeGeoKey', 3073: 'PCSCitationGeoKey', 3074: 'ProjectionGeoKey', 3075: 'ProjCoordTransGeoKey', 3076: 'ProjLinearUnitsGeoKey', 3077: 'ProjLinearUnitSizeGeoKey', 3078: 'ProjStdParallel1GeoKey', 3079: 'ProjStdParallel2GeoKey', 3080: 'ProjNatOriginLongGeoKey', 3081: 'ProjNatOriginLatGeoKey', 3082: 'ProjFalseEastingGeoKey', 3083: 'ProjFalseNorthingGeoKey', 3084: 'ProjFalseOriginLongGeoKey', 3085: 'ProjFalseOriginLatGeoKey', 3086: 'ProjFalseOriginEastingGeoKey', 3087: 'ProjFalseOriginNorthingGeoKey', 3088: 'ProjCenterLongGeoKey', 3089: 'ProjCenterLatGeoKey', 3090: 'ProjCenterEastingGeoKey', 3091: 'ProjFalseOriginNorthingGeoKey', 3092: 'ProjScaleAtNatOriginGeoKey', 3093: 'ProjScaleAtCenterGeoKey', 3094: 'ProjAzimuthAngleGeoKey', 3095: 'ProjStraightVertPoleLongGeoKey', 3096: 'ProjRectifiedGridAngleGeoKey', 4096: 'VerticalCSTypeGeoKey', 4097: 'VerticalCitationGeoKey', 4098: 'VerticalDatumGeoKey', 4099: 'VerticalUnitsGeoKey'}