import logging; log = logging.getLogger(__name__)
import sys
from passlib.utils.decor import deprecated_method
from abc import ABCMeta, abstractmethod, abstractproperty

        given a disabled-hash string,
        extract previously-enabled hash if one is present,
        otherwise raises ValueError
        