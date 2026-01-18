import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_trait_list_event(self):
    """ Record TraitListEvent behavior.
        """
    self.obj.alist = [1, 2, 3, 4]
    self.obj.on_trait_change(self._record_trait_list_event, 'alist_items')
    del self.obj.alist[0]
    self.assertLastTraitListEventEqual(0, [1], [])
    self.obj.alist.append(5)
    self.assertLastTraitListEventEqual(3, [], [5])
    self.obj.alist[0:2] = [6, 7]
    self.assertLastTraitListEventEqual(0, [2, 3], [6, 7])
    self.obj.alist[:2] = [4, 5]
    self.assertLastTraitListEventEqual(0, [6, 7], [4, 5])
    self.obj.alist[0:2:1] = [8, 9]
    self.assertLastTraitListEventEqual(0, [4, 5], [8, 9])
    self.obj.alist[0:2:1] = [8, 9]
    self.assertLastTraitListEventEqual(0, [8, 9], [8, 9])
    old_event = self.last_event
    self.obj.alist[4:] = []
    self.assertIs(self.last_event, old_event)
    self.obj.alist[0:4:2] = [10, 11]
    self.assertLastTraitListEventEqual(slice(0, 3, 2), [8, 4], [10, 11])
    del self.obj.alist[1:4:2]
    self.assertLastTraitListEventEqual(slice(1, 4, 2), [9, 5], [])
    self.obj.alist = [1, 2, 3, 4]
    del self.obj.alist[2:4]
    self.assertLastTraitListEventEqual(2, [3, 4], [])
    self.obj.alist[:0] = [5, 6, 7, 8]
    self.assertLastTraitListEventEqual(0, [], [5, 6, 7, 8])
    del self.obj.alist[:2]
    self.assertLastTraitListEventEqual(0, [5, 6], [])
    del self.obj.alist[0:2]
    self.assertLastTraitListEventEqual(0, [7, 8], [])
    del self.obj.alist[:]
    self.assertLastTraitListEventEqual(0, [1, 2], [])