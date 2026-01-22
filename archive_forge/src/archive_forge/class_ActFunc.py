import math
class ActFunc(object):
    """ "virtual base class" for activation functions

  """

    def __call__(self, x):
        return self.Eval(x)