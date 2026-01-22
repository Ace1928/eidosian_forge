import warnings as _warnings
import hashlib as _hashlib
def _current(self):
    """Return a hash object for the current state.

        To be used only internally with digest() and hexdigest().
        """
    if self._hmac:
        return self._hmac
    else:
        h = self._outer.copy()
        h.update(self._inner.digest())
        return h