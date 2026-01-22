import sys
class PyPy_repr:
    """
    Override repr for EntryPoint objects on PyPy to avoid __iter__ access.
    Ref #97, #102.
    """
    affected = hasattr(sys, 'pypy_version_info')

    def __compat_repr__(self):

        def make_param(name):
            value = getattr(self, name)
            return '{name}={value!r}'.format(**locals())
        params = ', '.join(map(make_param, self._fields))
        return 'EntryPoint({params})'.format(**locals())
    if affected:
        __repr__ = __compat_repr__
    del affected