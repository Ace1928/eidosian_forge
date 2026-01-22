import torch
from . import _Tensor, Tensor
from .reference import _dims, _enable_layers, llist, ltuple
@property
def _batchtensor(self):
    if self._batchtensor_data is None:
        with _enable_layers(self._levels):
            print('bt multiply fallback')
            self._batchtensor_data = self._lhs._batchtensor * self._rhs._batchtensor
    return self._batchtensor_data