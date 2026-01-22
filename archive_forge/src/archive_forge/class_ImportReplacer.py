from .errors import BzrError, InternalBzrError
class ImportReplacer(ScopeReplacer):
    """This is designed to replace only a portion of an import list.

    It will replace itself with a module, and then make children
    entries also ImportReplacer objects.

    At present, this only supports 'import foo.bar.baz' syntax.
    """
    __slots__ = ('_import_replacer_children', '_member', '_module_path')

    def __init__(self, scope, name, module_path, member=None, children={}):
        """Upon request import 'module_path' as the name 'module_name'.
        When imported, prepare children to also be imported.

        :param scope: The scope that objects should be imported into.
            Typically this is globals()
        :param name: The variable name. Often this is the same as the
            module_path. 'breezy'
        :param module_path: A list for the fully specified module path
            ['breezy', 'foo', 'bar']
        :param member: The member inside the module to import, often this is
            None, indicating the module is being imported.
        :param children: Children entries to be imported later.
            This should be a map of children specifications.
            ::

                {'foo':(['breezy', 'foo'], None,
                    {'bar':(['breezy', 'foo', 'bar'], None {})})
                }

        Examples::

            import foo => name='foo' module_path='foo',
                          member=None, children={}
            import foo.bar => name='foo' module_path='foo', member=None,
                              children={'bar':(['foo', 'bar'], None, {}}
            from foo import bar => name='bar' module_path='foo', member='bar'
                                   children={}
            from foo import bar, baz would get translated into 2 import
            requests. On for 'name=bar' and one for 'name=baz'
        """
        if member is not None and children:
            raise ValueError('Cannot supply both a member and children')
        object.__setattr__(self, '_import_replacer_children', children)
        object.__setattr__(self, '_member', member)
        object.__setattr__(self, '_module_path', module_path)
        cls = object.__getattribute__(self, '__class__')
        ScopeReplacer.__init__(self, scope=scope, name=name, factory=cls._import)

    def _import(self, scope, name):
        children = object.__getattribute__(self, '_import_replacer_children')
        member = object.__getattribute__(self, '_member')
        module_path = object.__getattribute__(self, '_module_path')
        name = '.'.join(module_path)
        if member is not None:
            module = _builtin_import(name, scope, scope, [member], level=0)
            return getattr(module, member)
        else:
            module = _builtin_import(name, scope, scope, [], level=0)
            for path in module_path[1:]:
                module = getattr(module, path)
        for child_name, (child_path, child_member, grandchildren) in children.items():
            cls = object.__getattribute__(self, '__class__')
            cls(module.__dict__, name=child_name, module_path=child_path, member=child_member, children=grandchildren)
        return module