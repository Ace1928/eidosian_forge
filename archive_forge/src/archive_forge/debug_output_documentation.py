from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
Generic debug output for object that lists its property keys and values.

  Use as function:
  x = X()
  generic_repr(x)
  # X(y='hi', z=1)  # ID: 140030868127696

  Use as addition to class:
  class X:
    def __init__(self, y='hi'):
      self.y = y
      self.z = 1

    def __repr__(self):
      return generic_repr(self)
  # X(y='hi', z=1)  # ID: 140030868127696

  Note: Declaring a class by running eval on this repr output will work
  only if all properties are settable in the class's __init__. Nested objects
  may also cause an issue.

  Args:
    class_instance (object): Instance of class to print debug output for.

  Returns:
    Debug output string.
  