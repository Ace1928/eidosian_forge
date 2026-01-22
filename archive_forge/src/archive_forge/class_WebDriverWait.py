import time
import typing
from typing import Callable
from typing import Generic
from typing import Literal
from typing import TypeVar
from typing import Union
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.types import WaitExcTypes
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
class WebDriverWait(Generic[D]):

    def __init__(self, driver: D, timeout: float, poll_frequency: float=POLL_FREQUENCY, ignored_exceptions: typing.Optional[WaitExcTypes]=None):
        """Constructor, takes a WebDriver instance and timeout in seconds.

        :Args:
         - driver - Instance of WebDriver (Ie, Firefox, Chrome or Remote) or a WebElement
         - timeout - Number of seconds before timing out
         - poll_frequency - sleep interval between calls
           By default, it is 0.5 second.
         - ignored_exceptions - iterable structure of exception classes ignored during calls.
           By default, it contains NoSuchElementException only.

        Example::

         from selenium.webdriver.support.wait import WebDriverWait 

         element = WebDriverWait(driver, 10).until(lambda x: x.find_element(By.ID, "someId")) 

         is_disappeared = WebDriverWait(driver, 30, 1, (ElementNotVisibleException)).\\ 

                     until_not(lambda x: x.find_element(By.ID, "someId").is_displayed())
        """
        self._driver = driver
        self._timeout = float(timeout)
        self._poll = poll_frequency
        if self._poll == 0:
            self._poll = POLL_FREQUENCY
        exceptions = list(IGNORED_EXCEPTIONS)
        if ignored_exceptions:
            try:
                exceptions.extend(iter(ignored_exceptions))
            except TypeError:
                exceptions.append(ignored_exceptions)
        self._ignored_exceptions = tuple(exceptions)

    def __repr__(self):
        return f'<{type(self).__module__}.{type(self).__name__} (session="{self._driver.session_id}")>'

    def until(self, method: Callable[[D], Union[Literal[False], T]], message: str='') -> T:
        """Calls the method provided with the driver as an argument until the         return value does not evaluate to ``False``.

        :param method: callable(WebDriver)
        :param message: optional message for :exc:`TimeoutException`
        :returns: the result of the last call to `method`
        :raises: :exc:`selenium.common.exceptions.TimeoutException` if timeout occurs
        """
        screen = None
        stacktrace = None
        end_time = time.monotonic() + self._timeout
        while True:
            try:
                value = method(self._driver)
                if value:
                    return value
            except self._ignored_exceptions as exc:
                screen = getattr(exc, 'screen', None)
                stacktrace = getattr(exc, 'stacktrace', None)
            time.sleep(self._poll)
            if time.monotonic() > end_time:
                break
        raise TimeoutException(message, screen, stacktrace)

    def until_not(self, method: Callable[[D], T], message: str='') -> Union[T, Literal[True]]:
        """Calls the method provided with the driver as an argument until the         return value evaluates to ``False``.

        :param method: callable(WebDriver)
        :param message: optional message for :exc:`TimeoutException`
        :returns: the result of the last call to `method`, or
                  ``True`` if `method` has raised one of the ignored exceptions
        :raises: :exc:`selenium.common.exceptions.TimeoutException` if timeout occurs
        """
        end_time = time.monotonic() + self._timeout
        while True:
            try:
                value = method(self._driver)
                if not value:
                    return value
            except self._ignored_exceptions:
                return True
            time.sleep(self._poll)
            if time.monotonic() > end_time:
                break
        raise TimeoutException(message)