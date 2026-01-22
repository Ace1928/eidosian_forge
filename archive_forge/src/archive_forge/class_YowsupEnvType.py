import abc
import logging
from six import with_metaclass
class YowsupEnvType(abc.ABCMeta):

    def __init__(cls, name, bases, dct):
        if name != 'YowsupEnv':
            YowsupEnv.registerEnv(cls)
        super(YowsupEnvType, cls).__init__(name, bases, dct)