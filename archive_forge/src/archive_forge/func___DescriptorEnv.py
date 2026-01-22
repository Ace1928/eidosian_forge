import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
@contextlib.contextmanager
def __DescriptorEnv(self, message_descriptor):
    previous_env = self.__current_env
    self.__current_path.append(message_descriptor.name)
    self.__current_env = message_descriptor
    yield
    self.__current_path.pop()
    self.__current_env = previous_env