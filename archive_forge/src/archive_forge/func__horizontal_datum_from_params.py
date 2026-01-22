import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _horizontal_datum_from_params(cf_params):
    datum_name = cf_params.get('horizontal_datum_name')
    if datum_name and datum_name not in ('undefined', 'unknown'):
        try:
            return Datum.from_name(datum_name)
        except CRSError:
            pass
    ellipsoid = None
    ellipsoid_name = cf_params.get('reference_ellipsoid_name')
    try:
        ellipsoid = CustomEllipsoid(name=ellipsoid_name or 'undefined', semi_major_axis=cf_params.get('semi_major_axis'), semi_minor_axis=cf_params.get('semi_minor_axis'), inverse_flattening=cf_params.get('inverse_flattening'), radius=cf_params.get('earth_radius'))
    except CRSError:
        if ellipsoid_name and ellipsoid_name not in ('undefined', 'unknown'):
            ellipsoid = Ellipsoid.from_name(ellipsoid_name)
    prime_meridian = None
    prime_meridian_name = cf_params.get('prime_meridian_name')
    try:
        prime_meridian = CustomPrimeMeridian(name=prime_meridian_name or 'undefined', longitude=cf_params['longitude_of_prime_meridian'])
    except KeyError:
        if prime_meridian_name and prime_meridian_name not in ('undefined', 'unknown'):
            prime_meridian = PrimeMeridian.from_name(prime_meridian_name)
    if ellipsoid or prime_meridian:
        return CustomDatum(name=datum_name or 'undefined', ellipsoid=ellipsoid or 'WGS 84', prime_meridian=prime_meridian or 'Greenwich')
    return None