import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
def ungrab(self, class_instance):
    """Ungrab a previously grabbed motion event.
        """
    class_instance = weakref.ref(class_instance.__self__)
    if self.grab_exclusive_class == class_instance:
        self.grab_exclusive_class = None
    if class_instance in self.grab_list:
        self.grab_list.remove(class_instance)