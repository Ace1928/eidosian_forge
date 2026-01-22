from __future__ import annotations
import enum
class GCSE(enum.IntEnum):
    """Unspecified GCS based on ellipsoid."""
    Undefined = 0
    User_Defined = 32767
    Airy1830 = 4001
    AiryModified1849 = 4002
    AustralianNationalSpheroid = 4003
    Bessel1841 = 4004
    BesselModified = 4005
    BesselNamibia = 4006
    Clarke1858 = 4007
    Clarke1866 = 4008
    Clarke1866Michigan = 4009
    Clarke1880_Benoit = 4010
    Clarke1880_IGN = 4011
    Clarke1880_RGS = 4012
    Clarke1880_Arc = 4013
    Clarke1880_SGA1922 = 4014
    Everest1830_1937Adjustment = 4015
    Everest1830_1967Definition = 4016
    Everest1830_1975Definition = 4017
    Everest1830Modified = 4018
    GRS1980 = 4019
    Helmert1906 = 4020
    IndonesianNationalSpheroid = 4021
    International1924 = 4022
    International1967 = 4023
    Krassowsky1940 = 4024
    NWL9D = 4025
    NWL10D = 4026
    Plessis1817 = 4027
    Struve1860 = 4028
    WarOffice = 4029
    WGS84 = 4030
    GEM10C = 4031
    OSU86F = 4032
    OSU91A = 4033
    Clarke1880 = 4034
    Sphere = 4035