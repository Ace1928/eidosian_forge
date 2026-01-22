import abc
import io
import itertools
from typing import BinaryIO, List
from .abc import Traversable, TraversableResources
class TraversableReader(TraversableResources, SimpleReader):
    """
    A TraversableResources based on SimpleReader. Resource providers
    may derive from this class to provide the TraversableResources
    interface by supplying the SimpleReader interface.
    """

    def files(self):
        return ResourceContainer(self)