import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
class LinkComponents(list):

    def add(self, c):
        component = c.component()
        self.append(component)
        return component