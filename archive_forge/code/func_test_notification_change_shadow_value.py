import pickle
import unittest
from traits.api import HasTraits, Int, List, Map, on_trait_change, TraitError
def test_notification_change_shadow_value(self):

    class PreferencesWithDynamicDefault(Preferences):

        def _color_default(self):
            return 'yellow'
    preferences = PreferencesWithDynamicDefault()
    self.assertEqual(len(preferences.primary_changes), 0)
    self.assertEqual(len(preferences.shadow_changes), 0)
    preferences.color_
    self.assertEqual(len(preferences.primary_changes), 0)
    self.assertEqual(len(preferences.shadow_changes), 0)