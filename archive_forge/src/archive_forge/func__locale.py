import logging
import os.path
import sys
from .exceptions import NoSuchClassError, UnsupportedPropertyError
from .icon_cache import IconCache
def _locale(self, prop):
    lang = getattr(QtCore.QLocale, prop.attrib['language'])
    country = getattr(QtCore.QLocale, prop.attrib['country'])
    return QtCore.QLocale(lang, country)