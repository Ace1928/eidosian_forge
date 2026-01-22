import typing
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions
class _SafariOptionsDescriptor:
    """_SafariOptionsDescriptor is an implementation of Descriptor protocol:

    : Any look-up or assignment to the below attributes in `Options` class will be intercepted
    by `__get__` and `__set__` method respectively.

    - `automatic_inspection`
    - `automatic_profiling`
    - `use_technology_preview`

    : When an attribute lookup happens,
    Example:
        `self.automatic_inspection`
        `__get__` method does a dictionary look up in the dictionary `_caps` of `Options` class
        and returns the value of key `safari:automaticInspection`
    : When an attribute assignment happens,
    Example:
        `self.automatic_inspection` = True
        `__set__` method sets/updates the value of the key `safari:automaticInspection` in `_caps`
        dictionary in `Options` class.
    """

    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, obj, cls):
        if self.name == 'Safari Technology Preview':
            return obj._caps.get('browserName') == self.name
        return obj._caps.get(self.name)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(f'{self.name} must be of type {self.expected_type}')
        if self.name == 'Safari Technology Preview':
            obj._caps['browserName'] = self.name if value else 'safari'
        else:
            obj._caps[self.name] = value