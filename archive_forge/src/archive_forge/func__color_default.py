import pickle
import unittest
from traits.api import HasTraits, Int, List, Map, on_trait_change, TraitError
def _color_default(self):
    return 'yellow'