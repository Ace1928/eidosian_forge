from collections import namedtuple
def _calculate_zero_point(self):
    scaled_value = self._w_max * (1 - self._beta) / self._scale_constant
    k = scaled_value ** (1 / 3.0)
    return k