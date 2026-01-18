import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
 BaseDirectory with no existence check accepts any pathlib path.
        