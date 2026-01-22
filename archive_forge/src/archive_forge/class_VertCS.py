from __future__ import annotations
import enum
class VertCS(enum.IntEnum):
    """Vertical CS Type Codes."""
    Undefined = 0
    User_Defined = 32767
    Airy_1830_ellipsoid = 5001
    Airy_Modified_1849_ellipsoid = 5002
    ANS_ellipsoid = 5003
    Bessel_1841_ellipsoid = 5004
    Bessel_Modified_ellipsoid = 5005
    Bessel_Namibia_ellipsoid = 5006
    Clarke_1858_ellipsoid = 5007
    Clarke_1866_ellipsoid = 5008
    Clarke_1880_Benoit_ellipsoid = 5010
    Clarke_1880_IGN_ellipsoid = 5011
    Clarke_1880_RGS_ellipsoid = 5012
    Clarke_1880_Arc_ellipsoid = 5013
    Clarke_1880_SGA_1922_ellipsoid = 5014
    Everest_1830_1937_Adjustment_ellipsoid = 5015
    Everest_1830_1967_Definition_ellipsoid = 5016
    Everest_1830_1975_Definition_ellipsoid = 5017
    Everest_1830_Modified_ellipsoid = 5018
    GRS_1980_ellipsoid = 5019
    Helmert_1906_ellipsoid = 5020
    INS_ellipsoid = 5021
    International_1924_ellipsoid = 5022
    International_1967_ellipsoid = 5023
    Krassowsky_1940_ellipsoid = 5024
    NWL_9D_ellipsoid = 5025
    NWL_10D_ellipsoid = 5026
    Plessis_1817_ellipsoid = 5027
    Struve_1860_ellipsoid = 5028
    War_Office_ellipsoid = 5029
    WGS_84_ellipsoid = 5030
    GEM_10C_ellipsoid = 5031
    OSU86F_ellipsoid = 5032
    OSU91A_ellipsoid = 5033
    Newlyn = 5101
    North_American_Vertical_Datum_1929 = 5102
    North_American_Vertical_Datum_1988 = 5103
    Yellow_Sea_1956 = 5104
    Baltic_Sea = 5105
    Caspian_Sea = 5106