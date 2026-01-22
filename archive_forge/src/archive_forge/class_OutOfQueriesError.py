import ipaddress
from typing import Optional, Union
class OutOfQueriesError(GeoIP2Error):
    """Your account is out of funds for the service queried."""