import typing
from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.common.proxy import Proxy
class _UnHandledPromptBehaviorDescriptor:
    """How the driver should respond when an alert is present and the:
    command sent is not handling the alert:
    https://w3c.github.io/webdriver/#dfn-table-of-page-load-strategies:

    :param behavior: behavior to use when an alert is encountered

    :returns: Values for implicit timeout, pageLoad timeout and script timeout if set (in milliseconds)
    """

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls):
        return obj._caps.get(self.name)

    def __set__(self, obj, value):
        if value in ('dismiss', 'accept', 'dismiss and notify', 'accept and notify', 'ignore'):
            obj.set_capability(self.name, value)
        else:
            raise ValueError('Behavior can only be one of the following: dismiss, accept, dismiss and notify, accept and notify, ignore')