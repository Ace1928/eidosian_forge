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
def url_matches(pattern: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking the current url.

    pattern is the expected pattern.  This finds the first occurrence of
    pattern in the current url and as such does not require an exact
    full match.
    """

    def _predicate(driver: WebDriver):
        return re.search(pattern, driver.current_url) is not None
    return _predicate