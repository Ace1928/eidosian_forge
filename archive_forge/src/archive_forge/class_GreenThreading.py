import sys
from futurist import _utils
class GreenThreading(object):

    @staticmethod
    def event_object(*args, **kwargs):
        return greenthreading.Event(*args, **kwargs)

    @staticmethod
    def lock_object(*args, **kwargs):
        return greenthreading.Lock(*args, **kwargs)

    @staticmethod
    def rlock_object(*args, **kwargs):
        return greenthreading.RLock(*args, **kwargs)

    @staticmethod
    def condition_object(*args, **kwargs):
        return greenthreading.Condition(*args, **kwargs)