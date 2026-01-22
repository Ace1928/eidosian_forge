import sys
class classproperty(object):
    """Class property decorator.

  Example usage:

  class MyClass(object):

    @classproperty
    def value(cls):
      return '123'

  > print MyClass.value
  123
  """

    def __init__(self, func):
        self._func = func

    def __get__(self, owner_self, owner_cls):
        return self._func(owner_cls)