import ipaddress
from abc import ABCMeta
from typing import Dict, List, Optional, Type, Union
from geoip2.mixins import SimpleEquality
class Subdivisions(tuple):
    """A tuple-like collection of subdivisions associated with an IP address.

    This class contains the subdivisions of the country associated with the
    IP address from largest to smallest.

    For instance, the response for Oxford in the United Kingdom would have
    England as the first element and Oxfordshire as the second element.

    This attribute is returned by ``city``, ``enterprise``, and ``insights``.
    """

    def __new__(cls: Type['Subdivisions'], locales: Optional[List[str]], *subdivisions) -> 'Subdivisions':
        subobjs = tuple((Subdivision(locales, **x) for x in subdivisions))
        obj = super().__new__(cls, subobjs)
        return obj

    def __init__(self, locales: Optional[List[str]], *subdivisions) -> None:
        self._locales = locales
        super().__init__()

    @property
    def most_specific(self) -> Subdivision:
        """The most specific (smallest) subdivision available.

        If there are no :py:class:`Subdivision` objects for the response,
        this returns an empty :py:class:`Subdivision`.

        :type: :py:class:`Subdivision`
        """
        try:
            return self[-1]
        except IndexError:
            return Subdivision(self._locales)