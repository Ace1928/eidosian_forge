import re
class value_property(object):
    """
    Represents a property that has a value in the Cache-Control header.

    When no value is actually given, the value of self.none is returned.
    """

    def __init__(self, prop, default=None, none=None, type=None):
        self.prop = prop
        self.default = default
        self.none = none
        self.type = type

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        if self.prop in obj.properties:
            value = obj.properties[self.prop]
            if value is None:
                return self.none
            else:
                return value
        else:
            return self.default

    def __set__(self, obj, value):
        if self.type is not None and self.type != obj.type:
            raise AttributeError('The property %s only applies to %s Cache-Control' % (self.prop, self.type))
        if value == self.default:
            if self.prop in obj.properties:
                del obj.properties[self.prop]
        elif value is True:
            obj.properties[self.prop] = None
        else:
            obj.properties[self.prop] = value

    def __delete__(self, obj):
        if self.prop in obj.properties:
            del obj.properties[self.prop]