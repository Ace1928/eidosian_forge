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
def visibility_of_all_elements_located(locator: Tuple[str, str]) -> Callable[[WebDriverOrWebElement], Union[List[WebElement], Literal[False]]]:
    """An expectation for checking that all elements are present on the DOM of
    a page and visible. Visibility means that the elements are not only
    displayed but also has a height and width that is greater than 0.

    locator - used to find the elements
    returns the list of WebElements once they are located and visible
    """

    def _predicate(driver: WebDriverOrWebElement):
        try:
            elements = driver.find_elements(*locator)
            for element in elements:
                if _element_if_visible(element, visibility=False):
                    return False
            return elements
        except StaleElementReferenceException:
            return False
    return _predicate