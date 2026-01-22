from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _NextItem(self):
    """Returns the next item in self._iterable."""
    if self._injected:
        self._injected = False
        return self._injected_value
    try:
        return next(self._iterable)
    except TypeError:
        pass
    except StopIteration:
        self._tap.Done()
        raise
    try:
        return self._iterable.pop(0)
    except (AttributeError, KeyError, TypeError):
        pass
    except IndexError:
        self._tap.Done()
        raise StopIteration
    if self._iterable is None or self._stop:
        self._tap.Done()
        raise StopIteration
    self._stop = True
    return self._iterable