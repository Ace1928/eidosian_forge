import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def add_name_test(self) -> None:
    if self.element == '*':
        return
    self.add_condition('name() = %s' % GenericTranslator.xpath_literal(self.element))
    self.element = '*'