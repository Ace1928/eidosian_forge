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
def title_is(title: str) -> Callable[[WebDriver], bool]:
    """An expectation for checking the title of a page.

    title is the expected title, which must be an exact match returns
    True if the title matches, false otherwise.
    """

    def _predicate(driver: WebDriver):
        return driver.title == title
    return _predicate