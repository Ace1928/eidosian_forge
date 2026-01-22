import numpy as np
from scipy import stats  #get rid of this? need only norm.sf
class ResultsBunch(dict):
    template = '%r'

    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__ = self
        self._initialize()

    def _initialize(self):
        pass

    def __str__(self):
        return self.template % self