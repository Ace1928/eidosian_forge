import sys
from .string import StrPrinter
from .numbers import number_to_scientific_unicode
def unicode_(obj, **settings):
    return UnicodePrinter(settings).doprint(obj)