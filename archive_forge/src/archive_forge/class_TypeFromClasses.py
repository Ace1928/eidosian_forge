import inspect
from ast import literal_eval
from traitlets import Any, ClassBasedTraitType, TraitError, Undefined
from traitlets.utils.descriptions import describe
class TypeFromClasses(ClassBasedTraitType):
    """A trait whose value must be a subclass of a class in a specified list of classes."""
    default_value: Any

    def __init__(self, default_value=Undefined, klasses=None, **kwargs):
        """Construct a Type trait
        A Type trait specifies that its values must be subclasses of
        a class in a list of possible classes.
        If only ``default_value`` is given, it is used for the ``klasses`` as
        well. If neither are given, both default to ``object``.
        Parameters
        ----------
        default_value : class, str or None
            The default value must be a subclass of klass.  If an str,
            the str must be a fully specified class name, like 'foo.bar.Bah'.
            The string is resolved into real class, when the parent
            :class:`HasTraits` class is instantiated.
        klasses : list of class, str [ default object ]
            Values of this trait must be a subclass of klass.  The klass
            may be specified in a string like: 'foo.bar.MyClass'.
            The string is resolved into real class, when the parent
            :class:`HasTraits` class is instantiated.
        allow_none : bool [ default False ]
            Indicates whether None is allowed as an assignable value.
        """
        if default_value is Undefined:
            new_default_value = object if klasses is None else klasses
        else:
            new_default_value = default_value
        if klasses is None:
            if default_value is None or default_value is Undefined:
                klasses = [object]
            else:
                klasses = [default_value]
        if not isinstance(klasses, (list, tuple, set)):
            msg = '`klasses` must be a list of class names (type is str) or classes.'
            raise TraitError(msg)
        for klass in klasses:
            if not (inspect.isclass(klass) or isinstance(klass, str)):
                msg = 'A OneOfType trait must specify a list of classes.'
                raise TraitError(msg)
        self.klasses = klasses
        super().__init__(new_default_value, **kwargs)

    def subclass_from_klasses(self, value):
        """Check that a given class is a subclasses found in the klasses list."""
        return any((issubclass(value, klass) for klass in self.importable_klasses))

    def validate(self, obj, value):
        """Validates that the value is a valid object instance."""
        if isinstance(value, str):
            try:
                value = self._resolve_string(value)
            except ImportError as e:
                emsg = f"The '{self.name}' trait of {obj} instance must be a type, but {value!r} could not be imported"
                raise TraitError(emsg) from e
        try:
            if self.subclass_from_klasses(value):
                return value
        except Exception:
            pass
        self.error(obj, value)

    def info(self):
        """Returns a description of the trait."""
        result = 'a subclass of '
        for klass in self.klasses:
            if not isinstance(klass, str):
                klass = klass.__module__ + '.' + klass.__name__
            result += f'{klass} or '
        result = result.strip(' or ')
        if self.allow_none:
            return result + ' or None'
        return result

    def instance_init(self, obj):
        """Initialize an instance."""
        self._resolve_classes()
        super().instance_init(obj)

    def _resolve_classes(self):
        """Resolve all string names to actual classes."""
        self.importable_klasses = []
        for klass in self.klasses:
            if isinstance(klass, str):
                try:
                    klass = self._resolve_string(klass)
                    self.importable_klasses.append(klass)
                except Exception:
                    pass
            else:
                self.importable_klasses.append(klass)
        if isinstance(self.default_value, str):
            self.default_value = self._resolve_string(self.default_value)

    def default_value_repr(self):
        """The default value repr."""
        value = self.default_value
        if isinstance(value, str):
            return repr(value)
        else:
            return repr(f'{value.__module__}.{value.__name__}')