import datetime
import numpy as np
import shapely.geometry as sgeom
from .. import crs as ccrs
from . import ShapelyFeature
def _solar_position(date):
    """
    Calculate the latitude and longitude point where the sun is
    directly overhead for the given date.

    Parameters
    ----------
    date
        A UTC datetime object.

    Returns
    -------
    (latitude, longitude) in degrees

    Note
    ----
    Algorithm implemented following equations from Chapter 5 (Algorithm 29):
    Vallado, David 'Fundamentals of Astrodynamics and Applications', (2007)

    """
    T_UT1 = (_julian_day(date) - 2451545.0) / 36525
    lambda_M_sun = (280.46 + 36000.771 * T_UT1) % 360
    M_sun = (357.5277233 + 35999.05034 * T_UT1) % 360
    lambda_ecliptic = lambda_M_sun + 1.914666471 * np.sin(np.deg2rad(M_sun)) + 0.019994643 * np.sin(np.deg2rad(2 * M_sun))
    epsilon = 23.439291 - 0.0130042 * T_UT1
    delta_sun = np.rad2deg(np.arcsin(np.sin(np.deg2rad(epsilon)) * np.sin(np.deg2rad(lambda_ecliptic))))
    theta_GMST = 67310.54841 + (876600 * 3600 + 8640184.812866) * T_UT1 + 0.093104 * T_UT1 ** 2 - 6.2e-06 * T_UT1 ** 3
    theta_GMST = theta_GMST % 86400 / 240
    numerator = np.cos(np.deg2rad(epsilon)) * np.sin(np.deg2rad(lambda_ecliptic)) / np.cos(np.deg2rad(delta_sun))
    denominator = np.cos(np.deg2rad(lambda_ecliptic)) / np.cos(np.deg2rad(delta_sun))
    alpha_sun = np.rad2deg(np.arctan2(numerator, denominator))
    lon = -(theta_GMST - alpha_sun)
    if lon < -180:
        lon += 360
    return (delta_sun, lon)