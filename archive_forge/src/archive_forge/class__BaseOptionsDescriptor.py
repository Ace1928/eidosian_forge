import typing
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.common.proxy import Proxy
class _BaseOptionsDescriptor:

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls):
        if self.name in ('acceptInsecureCerts', 'strictFileInteractability', 'setWindowRect', 'se:downloadsEnabled'):
            return obj._caps.get(self.name, False)
        return obj._caps.get(self.name)

    def __set__(self, obj, value):
        obj.set_capability(self.name, value)