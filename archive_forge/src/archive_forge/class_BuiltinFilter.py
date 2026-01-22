from __future__ import annotations
import functools
class BuiltinFilter(MultibandFilter):

    def filter(self, image):
        if image.mode == 'P':
            msg = 'cannot filter palette images'
            raise ValueError(msg)
        return image.filter(*self.filterargs)