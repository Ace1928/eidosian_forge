from enum import Enum
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
@property
def additional_options(self) -> dict:
    """:Returns: The additional options."""
    return self._additional