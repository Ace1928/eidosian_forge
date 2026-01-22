from decimal import Decimal
from boto.compat import filter, map
def _declared(self, op, **kw):

    def inherit(obj):
        result = {}
        for cls in getattr(obj, '__bases__', ()):
            result.update(inherit(cls))
        result.update(obj.__dict__)
        return result
    scope = inherit(self.__class__)
    scope.update(self.__dict__)
    declared = lambda attr: isinstance(attr[1], DeclarativeType)
    for name, node in filter(declared, scope.items()):
        getattr(node, op)(self, name, parentname=self._name, **kw)