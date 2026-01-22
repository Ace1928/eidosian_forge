from typing import Dict
import dns.exception
from dns._asyncbackend import (  # noqa: F401  lgtm[py/unused-import]
class AsyncLibraryNotFoundError(dns.exception.DNSException):
    pass