from OpenGL import acceleratesupport
class Curry(object):
    """Provides a simple Curry which can bind (only) the first element

        This is used by lazywrapper, which explains the weird naming
        of the two attributes...
        """
    wrapperFunction = None
    baseFunction = None

    def __init__(self, wrapperFunction, baseFunction):
        """Stores self.wrapperFunction and self.baseFunction"""
        self.baseFunction = baseFunction
        self.wrapperFunction = wrapperFunction

    def __call__(self, *args, **named):
        """returns self.wrapperFunction( self.baseFunction, *args, **named )"""
        return self.wrapperFunction(self.baseFunction, *args, **named)