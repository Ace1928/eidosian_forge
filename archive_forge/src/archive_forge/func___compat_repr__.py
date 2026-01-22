import sys
def __compat_repr__(self):

    def make_param(name):
        value = getattr(self, name)
        return '{name}={value!r}'.format(**locals())
    params = ', '.join(map(make_param, self._fields))
    return 'EntryPoint({params})'.format(**locals())