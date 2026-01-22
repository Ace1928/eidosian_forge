import re
class UpdateDict(dict):
    """
    Dict that has a callback on all updates
    """
    updated = None
    updated_args = None

    def _updated(self):
        """
        Assign to new_dict.updated to track updates
        """
        updated = self.updated
        if updated is not None:
            args = self.updated_args
            if args is None:
                args = (self,)
            updated(*args)

    def __setitem__(self, key, item):
        dict.__setitem__(self, key, item)
        self._updated()

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._updated()

    def clear(self):
        dict.clear(self)
        self._updated()

    def update(self, *args, **kw):
        dict.update(self, *args, **kw)
        self._updated()

    def setdefault(self, key, value=None):
        val = dict.setdefault(self, key, value)
        if val is value:
            self._updated()
        return val

    def pop(self, *args):
        v = dict.pop(self, *args)
        self._updated()
        return v

    def popitem(self):
        v = dict.popitem(self)
        self._updated()
        return v