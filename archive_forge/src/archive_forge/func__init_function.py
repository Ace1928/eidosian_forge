import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform
def _init_function(self, r):
    if isinstance(self.function, str):
        self.function = self.function.lower()
        _mapped = {'inverse': 'inverse_multiquadric', 'inverse multiquadric': 'inverse_multiquadric', 'thin-plate': 'thin_plate'}
        if self.function in _mapped:
            self.function = _mapped[self.function]
        func_name = '_h_' + self.function
        if hasattr(self, func_name):
            self._function = getattr(self, func_name)
        else:
            functionlist = [x[3:] for x in dir(self) if x.startswith('_h_')]
            raise ValueError('function must be a callable or one of ' + ', '.join(functionlist))
        self._function = getattr(self, '_h_' + self.function)
    elif callable(self.function):
        allow_one = False
        if hasattr(self.function, 'func_code') or hasattr(self.function, '__code__'):
            val = self.function
            allow_one = True
        elif hasattr(self.function, '__call__'):
            val = self.function.__call__.__func__
        else:
            raise ValueError('Cannot determine number of arguments to function')
        argcount = val.__code__.co_argcount
        if allow_one and argcount == 1:
            self._function = self.function
        elif argcount == 2:
            self._function = self.function.__get__(self, Rbf)
        else:
            raise ValueError('Function argument must take 1 or 2 arguments.')
    a0 = self._function(r)
    if a0.shape != r.shape:
        raise ValueError('Callable must take array and return array of the same shape')
    return a0