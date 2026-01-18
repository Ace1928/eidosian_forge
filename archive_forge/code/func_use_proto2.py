import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
@property
def use_proto2(self):
    return self.__use_proto2