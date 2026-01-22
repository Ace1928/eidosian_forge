from .._cloudpickle_wrapper import wrap_non_picklable_objects
from .._cloudpickle_wrapper import _my_wrap_non_picklable_objects
class AClass(object):

    def __call__(self, x):
        return x