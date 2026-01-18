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
def url_to_be(url: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking the current url.

    url is the expected url, which must be an exact match returns True
    if the url matches, false otherwise.
    """

    def _predicate(driver: WebDriver):
        return url == driver.current_url
    return _predicate