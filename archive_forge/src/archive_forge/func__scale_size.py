from decimal import Decimal
from kivy.lang import Builder
from kivy.properties import ListProperty
def _scale_size(self, size, sizer):
    size_new = list(sizer)
    i = size_new.index(None)
    j = i * -1 + 1
    size_new[i] = size_new[j] * size[i] / size[j]
    return tuple(size_new)