def _alreadyPatched(self, obj, name):
    """
        Has the C{name} attribute of C{obj} already been patched by this
        patcher?
        """
    for o, n, v in self._originals:
        if (o, n) == (obj, name):
            return True
    return False