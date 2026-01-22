class PermWrapper:

    def __init__(self, user):
        self.user = user

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.user!r})'

    def __getitem__(self, app_label):
        return PermLookupDict(self.user, app_label)

    def __iter__(self):
        raise TypeError('PermWrapper is not iterable.')

    def __contains__(self, perm_name):
        """
        Lookup by "someapp" or "someapp.someperm" in perms.
        """
        if '.' not in perm_name:
            return bool(self[perm_name])
        app_label, perm_name = perm_name.split('.', 1)
        return self[app_label][perm_name]