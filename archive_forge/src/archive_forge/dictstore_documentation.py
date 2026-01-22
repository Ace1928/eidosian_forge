import errno
from os.path import exists, abspath, dirname
from kivy.compat import iteritems
from kivy.storage import AbstractStore
Store implementation using a pickled `dict`.
    See the :mod:`kivy.storage` module documentation for more information.
    