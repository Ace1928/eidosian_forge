import unittest
class PropertyWidget(EventDispatcher):
    foo = BoundedNumericProperty(1, min=-5, max=5)