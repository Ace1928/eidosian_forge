import pickle
import unittest
from traits.api import HasTraits, Int, List, Map, on_trait_change, TraitError
class Preferences(HasTraits):
    """
    Example class with a Map that records changes to that map.
    """
    primary_changes = List()
    shadow_changes = List()
    color = Map({'red': 4, 'green': 2, 'yellow': 6}, default_value='yellow')

    @on_trait_change('color')
    def _record_primary_trait_change(self, obj, name, old, new):
        change = (obj, name, old, new)
        self.primary_changes.append(change)

    @on_trait_change('color_')
    def _record_shadow_trait_change(self, obj, name, old, new):
        change = (obj, name, old, new)
        self.shadow_changes.append(change)