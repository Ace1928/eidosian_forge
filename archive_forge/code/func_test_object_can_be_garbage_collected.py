import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def test_object_can_be_garbage_collected(self):
    import weakref

    def listener():
        pass
    obj = DynamicNotifiers()
    obj.on_trait_change(listener, 'ok')
    obj_collected = []

    def obj_collected_callback(weakref):
        obj_collected.append(True)
    obj_weakref = weakref.ref(obj, obj_collected_callback)
    del obj
    self.assertEqual(obj_collected, [True])
    self.assertIsNone(obj_weakref())