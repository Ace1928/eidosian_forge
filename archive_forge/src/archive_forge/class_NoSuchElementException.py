from typing import Optional
from typing import Sequence
class NoSuchElementException(WebDriverException):
    """Thrown when element could not be found.

    If you encounter this exception, you may want to check the following:
        * Check your selector used in your find_by...
        * Element may not yet be on the screen at the time of the find operation,
          (webpage is still loading) see selenium.webdriver.support.wait.WebDriverWait()
          for how to write a wait wrapper to wait for an element to appear.
    """

    def __init__(self, msg: Optional[str]=None, screen: Optional[str]=None, stacktrace: Optional[Sequence[str]]=None) -> None:
        with_support = f'{msg}; {SUPPORT_MSG} {ERROR_URL}#no-such-element-exception'
        super().__init__(with_support, screen, stacktrace)