import copy
import re
class UnitRegistry:

    class __Registry:
        __shared_state = {}

        def __init__(self):
            self.__dict__ = self.__shared_state
            self.__context = {}

        def __getitem__(self, string):
            try:
                return eval(string, self.__context)
            except NameError:
                raise LookupError('Unable to parse units: "%s"' % string)

        def __setitem__(self, string, val):
            assert isinstance(string, str)
            try:
                assert string not in self.__context
            except AssertionError:
                if val == self.__context[string]:
                    return
                raise KeyError('%s has already been registered for %s' % (string, self.__context[string]))
            self.__context[string] = val
    __regex = re.compile('([A-Za-z])\\.([A-Za-z])')
    __registry = __Registry()

    def __getattr__(self, attr):
        return getattr(self.__registry, attr)

    def __setitem__(self, label, value):
        self.__registry.__setitem__(label, value)

    def __getitem__(self, label):
        """Parses a string description of a unit e.g., 'g/cc'"""
        label = self.__regex.sub('\\g<1>*\\g<2>', label.replace('^', '**').replace('·', '*'))
        if label == '':
            label = 'dimensionless'
        if '%' in label:
            label = label.replace('%', 'percent')
        if label.lower() == 'in':
            label = 'inch'
        return self.__registry[label]