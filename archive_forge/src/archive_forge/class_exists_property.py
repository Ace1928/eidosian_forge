import re
class exists_property(object):
    """
    Represents a property that either is listed in the Cache-Control
    header, or is not listed (has no value)
    """

    def __init__(self, prop, type=None):
        self.prop = prop
        self.type = type

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        return self.prop in obj.properties

    def __set__(self, obj, value):
        if self.type is not None and self.type != obj.type:
            raise AttributeError('The property %s only applies to %s Cache-Control' % (self.prop, self.type))
        if value:
            obj.properties[self.prop] = None
        elif self.prop in obj.properties:
            del obj.properties[self.prop]

    def __delete__(self, obj):
        self.__set__(obj, False)