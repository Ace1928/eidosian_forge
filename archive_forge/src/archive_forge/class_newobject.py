class newobject(object):
    """
    A magical object class that provides Python 2 compatibility methods::
        next
        __unicode__
        __nonzero__

    Subclasses of this class can merely define the Python 3 methods (__next__,
    __str__, and __bool__).

    This is a copy/paste of the future.types.newobject class of the future
    package.
    """

    def next(self):
        if hasattr(self, '__next__'):
            return type(self).__next__(self)
        raise TypeError('newobject is not an iterator')

    def __unicode__(self):
        if hasattr(self, '__str__'):
            s = type(self).__str__(self)
        else:
            s = str(self)
        if isinstance(s, unicode):
            return s
        else:
            return s.decode('utf-8')

    def __nonzero__(self):
        if hasattr(self, '__bool__'):
            return type(self).__bool__(self)
        return True

    def __long__(self):
        if not hasattr(self, '__int__'):
            return NotImplemented
        return self.__int__()

    def __native__(self):
        """
        Hook for the future.utils.native() function
        """
        return object(self)