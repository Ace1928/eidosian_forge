from typing import Optional
from typing import Union
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchFrameException
from selenium.common.exceptions import NoSuchWindowException
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from .command import Command
def _w3c_window(self, window_name: str) -> None:

    def send_handle(h):
        self._driver.execute(Command.SWITCH_TO_WINDOW, {'handle': h})
    try:
        send_handle(window_name)
    except NoSuchWindowException:
        original_handle = self._driver.current_window_handle
        handles = self._driver.window_handles
        for handle in handles:
            send_handle(handle)
            current_name = self._driver.execute_script('return window.name')
            if window_name == current_name:
                return
        send_handle(original_handle)
        raise