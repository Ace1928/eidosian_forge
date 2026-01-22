from .errors import BzrError, InternalBzrError
class ScopeReplacer:
    """A lazy object that will replace itself in the appropriate scope.

    This object sits, ready to create the real object the first time it is
    needed.
    """
    __slots__ = ('_scope', '_factory', '_name', '_real_obj')
    _should_proxy = True

    def __init__(self, scope, factory, name):
        """Create a temporary object in the specified scope.
        Once used, a real object will be placed in the scope.

        :param scope: The scope the object should appear in
        :param factory: A callable that will create the real object.
            It will be passed (self, scope, name)
        :param name: The variable name in the given scope.
        """
        object.__setattr__(self, '_scope', scope)
        object.__setattr__(self, '_factory', factory)
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_real_obj', None)
        scope[name] = self

    def _resolve(self):
        """Return the real object for which this is a placeholder"""
        name = object.__getattribute__(self, '_name')
        real_obj = object.__getattribute__(self, '_real_obj')
        if real_obj is None:
            factory = object.__getattribute__(self, '_factory')
            scope = object.__getattribute__(self, '_scope')
            obj = factory(self, scope, name)
            if obj is self:
                raise IllegalUseOfScopeReplacer(name, msg="Object tried to replace itself, check it's not using its own scope.")
            real_obj = object.__getattribute__(self, '_real_obj')
            if real_obj is None:
                object.__setattr__(self, '_real_obj', obj)
                scope[name] = obj
                return obj
        if not ScopeReplacer._should_proxy:
            raise IllegalUseOfScopeReplacer(name, msg='Object already replaced, did you assign it to another variable?')
        return real_obj

    def __getattribute__(self, attr):
        obj = object.__getattribute__(self, '_resolve')()
        return getattr(obj, attr)

    def __setattr__(self, attr, value):
        obj = object.__getattribute__(self, '_resolve')()
        return setattr(obj, attr, value)

    def __call__(self, *args, **kwargs):
        obj = object.__getattribute__(self, '_resolve')()
        return obj(*args, **kwargs)