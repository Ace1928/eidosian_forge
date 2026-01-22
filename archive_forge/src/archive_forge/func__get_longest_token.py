from typing import List
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
def _get_longest_token(self, value: str) -> str:
    items = value.split(' ')
    longest = ''
    for item in items:
        if len(item) > len(longest):
            longest = item
    return longest