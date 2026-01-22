from typing import Optional
from typing import Sequence
class InvalidCookieDomainException(WebDriverException):
    """Thrown when attempting to add a cookie under a different domain than the
    current URL."""