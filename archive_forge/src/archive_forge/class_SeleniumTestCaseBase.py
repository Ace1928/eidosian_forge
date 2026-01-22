import sys
import unittest
from contextlib import contextmanager
from django.test import LiveServerTestCase, tag
from django.utils.functional import classproperty
from django.utils.module_loading import import_string
from django.utils.text import capfirst
class SeleniumTestCaseBase(type(LiveServerTestCase)):
    browsers = []
    selenium_hub = None
    external_host = None
    browser = None
    headless = False

    def __new__(cls, name, bases, attrs):
        """
        Dynamically create new classes and add them to the test module when
        multiple browsers specs are provided (e.g. --selenium=firefox,chrome).
        """
        test_class = super().__new__(cls, name, bases, attrs)
        if test_class.browser or not any((name.startswith('test') and callable(value) for name, value in attrs.items())):
            return test_class
        elif test_class.browsers:
            first_browser = test_class.browsers[0]
            test_class.browser = first_browser
            host = test_class.host if not test_class.selenium_hub else '0.0.0.0'
            test_class.host = host
            test_class.external_host = cls.external_host
            module = sys.modules[test_class.__module__]
            for browser in test_class.browsers[1:]:
                browser_test_class = cls.__new__(cls, '%s%s' % (capfirst(browser), name), (test_class,), {'browser': browser, 'host': host, 'external_host': cls.external_host, '__module__': test_class.__module__})
                setattr(module, browser_test_class.__name__, browser_test_class)
            return test_class
        return unittest.skip('No browsers specified.')(test_class)

    @classmethod
    def import_webdriver(cls, browser):
        return import_string('selenium.webdriver.%s.webdriver.WebDriver' % browser)

    @classmethod
    def import_options(cls, browser):
        return import_string('selenium.webdriver.%s.options.Options' % browser)

    @classmethod
    def get_capability(cls, browser):
        from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
        return getattr(DesiredCapabilities, browser.upper())

    def create_options(self):
        options = self.import_options(self.browser)()
        if self.headless:
            match self.browser:
                case 'chrome':
                    options.add_argument('--headless=new')
                case 'firefox':
                    options.add_argument('-headless')
        return options

    def create_webdriver(self):
        options = self.create_options()
        if self.selenium_hub:
            from selenium import webdriver
            for key, value in self.get_capability(self.browser).items():
                options.set_capability(key, value)
            return webdriver.Remote(command_executor=self.selenium_hub, options=options)
        return self.import_webdriver(self.browser)(options=options)