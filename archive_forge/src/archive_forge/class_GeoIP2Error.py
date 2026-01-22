import ipaddress
from typing import Optional, Union
class GeoIP2Error(RuntimeError):
    """There was a generic error in GeoIP2.

    This class represents a generic error. It extends :py:exc:`RuntimeError`
    and does not add any additional attributes.

    """