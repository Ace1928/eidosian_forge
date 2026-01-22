import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class UTMConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the UTM conversion.

    :ref:`PROJ docs <utm>`
    """

    def __new__(cls, zone: str, hemisphere: str='N'):
        """
        Parameters
        ----------
        zone: int
            UTM Zone between 1-60.
        hemisphere: str, default="N"
            Either N for North or S for South.
        """
        return cls.from_name(f'UTM zone {zone}{hemisphere}')