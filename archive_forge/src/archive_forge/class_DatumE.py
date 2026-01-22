from __future__ import annotations
import enum
class DatumE(enum.IntEnum):
    """Ellipsoid-Only Geodetic Datum Codes."""
    Undefined = 0
    User_Defined = 32767
    Airy1830 = 6001
    AiryModified1849 = 6002
    AustralianNationalSpheroid = 6003
    Bessel1841 = 6004
    BesselModified = 6005
    BesselNamibia = 6006
    Clarke1858 = 6007
    Clarke1866 = 6008
    Clarke1866Michigan = 6009
    Clarke1880_Benoit = 6010
    Clarke1880_IGN = 6011
    Clarke1880_RGS = 6012
    Clarke1880_Arc = 6013
    Clarke1880_SGA1922 = 6014
    Everest1830_1937Adjustment = 6015
    Everest1830_1967Definition = 6016
    Everest1830_1975Definition = 6017
    Everest1830Modified = 6018
    GRS1980 = 6019
    Helmert1906 = 6020
    IndonesianNationalSpheroid = 6021
    International1924 = 6022
    International1967 = 6023
    Krassowsky1960 = 6024
    NWL9D = 6025
    NWL10D = 6026
    Plessis1817 = 6027
    Struve1860 = 6028
    WarOffice = 6029
    WGS84 = 6030
    GEM10C = 6031
    OSU86F = 6032
    OSU91A = 6033
    Clarke1880 = 6034
    Sphere = 6035