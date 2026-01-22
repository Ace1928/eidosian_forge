import re
from collections.abc import Iterable
from typing import Any
from typing import Callable
from typing import List
from typing import Literal
from typing import Tuple
from typing import TypeVar
from typing import Union
from selenium.common.exceptions import NoAlertPresentException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchFrameException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webdriver import WebElement
def alert_is_present() -> Callable[[WebDriver], Union[Alert, Literal[False]]]:
    """An expectation for checking if an alert is currently present and
    switching to it."""

    def _predicate(driver: WebDriver):
        try:
            return driver.switch_to.alert
        except NoAlertPresentException:
            return False
    return _predicate