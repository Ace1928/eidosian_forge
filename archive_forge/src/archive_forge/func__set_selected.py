from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def _set_selected(self, option) -> None:
    if not option.is_selected():
        if not option.is_enabled():
            raise NotImplementedError('You may not select a disabled option')
        option.click()