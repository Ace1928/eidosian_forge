from enum import Enum, IntFlag
class WktVersion(BaseEnum):
    """
     .. versionadded:: 2.2.0

    Supported CRS WKT string versions

    See: :c:enum:`PJ_WKT_TYPE`
    """
    WKT2_2015 = 'WKT2_2015'
    WKT2_2015_SIMPLIFIED = 'WKT2_2015_SIMPLIFIED'
    WKT2_2018 = 'WKT2_2018'
    WKT2_2018_SIMPLIFIED = 'WKT2_2018_SIMPLIFIED'
    WKT2_2019 = 'WKT2_2019'
    WKT2_2019_SIMPLIFIED = 'WKT2_2019_SIMPLIFIED'
    WKT1_GDAL = 'WKT1_GDAL'
    WKT1_ESRI = 'WKT1_ESRI'