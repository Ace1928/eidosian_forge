import logging
from typing import TYPE_CHECKING, List, Literal, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_driver(self) -> Union['Chrome', 'Firefox']:
    """Create and return a WebDriver instance based on the specified browser.

        Raises:
            ValueError: If an invalid browser is specified.

        Returns:
            Union[Chrome, Firefox]: A WebDriver instance for the specified browser.
        """
    if self.browser.lower() == 'chrome':
        from selenium.webdriver import Chrome
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.chrome.service import Service
        chrome_options = ChromeOptions()
        for arg in self.arguments:
            chrome_options.add_argument(arg)
        if self.headless:
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
        if self.binary_location is not None:
            chrome_options.binary_location = self.binary_location
        if self.executable_path is None:
            return Chrome(options=chrome_options)
        return Chrome(options=chrome_options, service=Service(executable_path=self.executable_path))
    elif self.browser.lower() == 'firefox':
        from selenium.webdriver import Firefox
        from selenium.webdriver.firefox.options import Options as FirefoxOptions
        from selenium.webdriver.firefox.service import Service
        firefox_options = FirefoxOptions()
        for arg in self.arguments:
            firefox_options.add_argument(arg)
        if self.headless:
            firefox_options.add_argument('--headless')
        if self.binary_location is not None:
            firefox_options.binary_location = self.binary_location
        if self.executable_path is None:
            return Firefox(options=firefox_options)
        return Firefox(options=firefox_options, service=Service(executable_path=self.executable_path))
    else:
        raise ValueError("Invalid browser specified. Use 'chrome' or 'firefox'.")