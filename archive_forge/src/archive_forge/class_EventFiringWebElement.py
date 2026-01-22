import typing
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from .abstract_event_listener import AbstractEventListener
class EventFiringWebElement:
    """A wrapper around WebElement instance which supports firing events."""

    def __init__(self, webelement: WebElement, ef_driver: EventFiringWebDriver) -> None:
        """Creates a new instance of the EventFiringWebElement."""
        self._webelement = webelement
        self._ef_driver = ef_driver
        self._driver = ef_driver.wrapped_driver
        self._listener = ef_driver._listener

    @property
    def wrapped_element(self) -> WebElement:
        """Returns the WebElement wrapped by this EventFiringWebElement
        instance."""
        return self._webelement

    def click(self) -> None:
        self._dispatch('click', (self._webelement, self._driver), 'click', ())

    def clear(self) -> None:
        self._dispatch('change_value_of', (self._webelement, self._driver), 'clear', ())

    def send_keys(self, *value) -> None:
        self._dispatch('change_value_of', (self._webelement, self._driver), 'send_keys', value)

    def find_element(self, by=By.ID, value=None) -> WebElement:
        return self._dispatch('find', (by, value, self._driver), 'find_element', (by, value))

    def find_elements(self, by=By.ID, value=None) -> typing.List[WebElement]:
        return self._dispatch('find', (by, value, self._driver), 'find_elements', (by, value))

    def _dispatch(self, l_call, l_args, d_call, d_args):
        getattr(self._listener, f'before_{l_call}')(*l_args)
        try:
            result = getattr(self._webelement, d_call)(*d_args)
        except Exception as exc:
            self._listener.on_exception(exc, self._driver)
            raise
        getattr(self._listener, f'after_{l_call}')(*l_args)
        return _wrap_elements(result, self._ef_driver)

    def __setattr__(self, item, value):
        if item.startswith('_') or not hasattr(self._webelement, item):
            object.__setattr__(self, item, value)
        else:
            try:
                object.__setattr__(self._webelement, item, value)
            except Exception as exc:
                self._listener.on_exception(exc, self._driver)
                raise

    def __getattr__(self, name):

        def _wrap(*args, **kwargs):
            try:
                result = attrib(*args, **kwargs)
                return _wrap_elements(result, self._ef_driver)
            except Exception as exc:
                self._listener.on_exception(exc, self._driver)
                raise
        try:
            attrib = getattr(self._webelement, name)
            return _wrap if callable(attrib) else attrib
        except Exception as exc:
            self._listener.on_exception(exc, self._driver)
            raise