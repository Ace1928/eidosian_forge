import array
import copy
import copyreg
import warnings
class _numpy_array(numpy.ndarray):

    def __deepcopy__(self, memo):
        """Overrides the deepcopy from numpy.ndarray that does not copy
            the object's attributes. This one will deepcopy the array and its
            :attr:`__dict__` attribute.
            """
        copy_ = numpy.ndarray.copy(self)
        copy_.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return copy_

    @staticmethod
    def __new__(cls, iterable):
        """Creates a new instance of a numpy.ndarray from a function call.
            Adds the possibility to instantiate from an iterable."""
        return numpy.array(list(iterable)).view(cls)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __reduce__(self):
        return (self.__class__, (list(self),), self.__dict__)