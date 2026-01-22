from abc import ABCMeta
from abc import abstractmethod
from ray.util.collective.types import (
@abstractmethod
def allreduce(self, tensor, allreduce_options=AllReduceOptions()):
    raise NotImplementedError()