import sys
import builtins as bltns
from types import MappingProxyType, DynamicClassAttribute
from operator import or_ as _or_
from functools import reduce
class EnumType(type):
    """
    Metaclass for Enum
    """

    @classmethod
    def __prepare__(metacls, cls, bases, **kwds):
        metacls._check_for_existing_members_(cls, bases)
        enum_dict = _EnumDict()
        enum_dict._cls_name = cls
        member_type, first_enum = metacls._get_mixins_(cls, bases)
        if first_enum is not None:
            enum_dict['_generate_next_value_'] = getattr(first_enum, '_generate_next_value_', None)
        return enum_dict

    def __new__(metacls, cls, bases, classdict, *, boundary=None, _simple=False, **kwds):
        if _simple:
            return super().__new__(metacls, cls, bases, classdict, **kwds)
        classdict.setdefault('_ignore_', []).append('_ignore_')
        ignore = classdict['_ignore_']
        for key in ignore:
            classdict.pop(key, None)
        member_names = classdict._member_names
        invalid_names = set(member_names) & {'mro', ''}
        if invalid_names:
            raise ValueError('invalid enum member name(s) %s' % ','.join((repr(n) for n in invalid_names)))
        _order_ = classdict.pop('_order_', None)
        classdict = dict(classdict.items())
        member_type, first_enum = metacls._get_mixins_(cls, bases)
        __new__, save_new, use_args = metacls._find_new_(classdict, member_type, first_enum)
        classdict['_new_member_'] = __new__
        classdict['_use_args_'] = use_args
        for name in member_names:
            value = classdict[name]
            classdict[name] = _proto_member(value)
        classdict['_member_names_'] = []
        classdict['_member_map_'] = {}
        classdict['_value2member_map_'] = {}
        classdict['_unhashable_values_'] = []
        classdict['_member_type_'] = member_type
        classdict['_value_repr_'] = metacls._find_data_repr_(cls, bases)
        classdict['_boundary_'] = boundary or getattr(first_enum, '_boundary_', None)
        classdict['_flag_mask_'] = 0
        classdict['_singles_mask_'] = 0
        classdict['_all_bits_'] = 0
        classdict['_inverted_'] = None
        try:
            exc = None
            enum_class = super().__new__(metacls, cls, bases, classdict, **kwds)
        except RuntimeError as e:
            exc = e.__cause__ or e
        if exc is not None:
            raise exc
        classdict.update(enum_class.__dict__)
        if ReprEnum is not None and ReprEnum in bases:
            if member_type is object:
                raise TypeError('ReprEnum subclasses must be mixed with a data type (i.e. int, str, float, etc.)')
            if '__format__' not in classdict:
                enum_class.__format__ = member_type.__format__
                classdict['__format__'] = enum_class.__format__
            if '__str__' not in classdict:
                method = member_type.__str__
                if method is object.__str__:
                    method = member_type.__repr__
                enum_class.__str__ = method
                classdict['__str__'] = enum_class.__str__
        for name in ('__repr__', '__str__', '__format__', '__reduce_ex__'):
            if name not in classdict:
                enum_method = getattr(first_enum, name)
                found_method = getattr(enum_class, name)
                object_method = getattr(object, name)
                data_type_method = getattr(member_type, name)
                if found_method in (data_type_method, object_method):
                    setattr(enum_class, name, enum_method)
        if Flag is not None and issubclass(enum_class, Flag):
            for name in ('__or__', '__and__', '__xor__', '__ror__', '__rand__', '__rxor__', '__invert__'):
                if name not in classdict:
                    enum_method = getattr(Flag, name)
                    setattr(enum_class, name, enum_method)
                    classdict[name] = enum_method
        if Enum is not None:
            if save_new:
                enum_class.__new_member__ = __new__
            enum_class.__new__ = Enum.__new__
        if _order_ is not None:
            if isinstance(_order_, str):
                _order_ = _order_.replace(',', ' ').split()
        if Flag is None and cls != 'Flag' or (Flag is not None and (not issubclass(enum_class, Flag))):
            delattr(enum_class, '_boundary_')
            delattr(enum_class, '_flag_mask_')
            delattr(enum_class, '_singles_mask_')
            delattr(enum_class, '_all_bits_')
            delattr(enum_class, '_inverted_')
        elif Flag is not None and issubclass(enum_class, Flag):
            member_list = [m._value_ for m in enum_class]
            if member_list != sorted(member_list):
                enum_class._iter_member_ = enum_class._iter_member_by_def_
            if _order_:
                _order_ = [o for o in _order_ if o not in enum_class._member_map_ or _is_single_bit(enum_class[o]._value_)]
        if _order_:
            _order_ = [o for o in _order_ if o not in enum_class._member_map_ or (o in enum_class._member_map_ and o in enum_class._member_names_)]
            if _order_ != enum_class._member_names_:
                raise TypeError('member order does not match _order_:\n  %r\n  %r' % (enum_class._member_names_, _order_))
        return enum_class

    def __bool__(cls):
        """
        classes/types should always be True.
        """
        return True

    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None):
        """
        Either returns an existing member, or creates a new enum class.

        This method is used both when an enum class is given a value to match
        to an enumeration member (i.e. Color(3)) and for the functional API
        (i.e. Color = Enum('Color', names='RED GREEN BLUE')).

        When used for the functional API:

        `value` will be the name of the new class.

        `names` should be either a string of white-space/comma delimited names
        (values will start at `start`), or an iterator/mapping of name, value pairs.

        `module` should be set to the module this class is being created in;
        if it is not set, an attempt to find that module will be made, but if
        it fails the class will not be picklable.

        `qualname` should be set to the actual location this class can be found
        at in its module; by default it is set to the global scope.  If this is
        not correct, unpickling will fail in some circumstances.

        `type`, if set, will be mixed in as the first base class.
        """
        if names is None:
            return cls.__new__(cls, value)
        return cls._create_(value, names, module=module, qualname=qualname, type=type, start=start, boundary=boundary)

    def __contains__(cls, member):
        """
        Return True if member is a member of this enum
        raises TypeError if member is not an enum member

        note: in 3.12 TypeError will no longer be raised, and True will also be
        returned if member is the value of a member in this enum
        """
        if not isinstance(member, Enum):
            import warnings
            warnings.warn('in 3.12 __contains__ will no longer raise TypeError, but will return True or\nFalse depending on whether the value is a member or the value of a member', DeprecationWarning, stacklevel=2)
            raise TypeError("unsupported operand type(s) for 'in': '%s' and '%s'" % (type(member).__qualname__, cls.__class__.__qualname__))
        return isinstance(member, cls) and member._name_ in cls._member_map_

    def __delattr__(cls, attr):
        if attr in cls._member_map_:
            raise AttributeError('%r cannot delete member %r.' % (cls.__name__, attr))
        super().__delattr__(attr)

    def __dir__(cls):
        interesting = set(['__class__', '__contains__', '__doc__', '__getitem__', '__iter__', '__len__', '__members__', '__module__', '__name__', '__qualname__'] + cls._member_names_)
        if cls._new_member_ is not object.__new__:
            interesting.add('__new__')
        if cls.__init_subclass__ is not object.__init_subclass__:
            interesting.add('__init_subclass__')
        if cls._member_type_ is object:
            return sorted(interesting)
        else:
            return sorted(set(dir(cls._member_type_)) | interesting)

    def __getattr__(cls, name):
        """
        Return the enum member matching `name`

        We use __getattr__ instead of descriptors or inserting into the enum
        class' __dict__ in order to support `name` and `value` being both
        properties for enum members (which live in the class' __dict__) and
        enum members themselves.
        """
        if _is_dunder(name):
            raise AttributeError(name)
        try:
            return cls._member_map_[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(cls, name):
        """
        Return the member matching `name`.
        """
        return cls._member_map_[name]

    def __iter__(cls):
        """
        Return members in definition order.
        """
        return (cls._member_map_[name] for name in cls._member_names_)

    def __len__(cls):
        """
        Return the number of members (no aliases)
        """
        return len(cls._member_names_)

    @bltns.property
    def __members__(cls):
        """
        Returns a mapping of member name->value.

        This mapping lists all enum members, including aliases. Note that this
        is a read-only view of the internal mapping.
        """
        return MappingProxyType(cls._member_map_)

    def __repr__(cls):
        if Flag is not None and issubclass(cls, Flag):
            return '<flag %r>' % cls.__name__
        else:
            return '<enum %r>' % cls.__name__

    def __reversed__(cls):
        """
        Return members in reverse definition order.
        """
        return (cls._member_map_[name] for name in reversed(cls._member_names_))

    def __setattr__(cls, name, value):
        """
        Block attempts to reassign Enum members.

        A simple assignment to the class namespace only changes one of the
        several possible ways to get an Enum member from the Enum class,
        resulting in an inconsistent Enumeration.
        """
        member_map = cls.__dict__.get('_member_map_', {})
        if name in member_map:
            raise AttributeError('cannot reassign member %r' % (name,))
        super().__setattr__(name, value)

    def _create_(cls, class_name, names, *, module=None, qualname=None, type=None, start=1, boundary=None):
        """
        Convenience method to create a new Enum class.

        `names` can be:

        * A string containing member names, separated either with spaces or
          commas.  Values are incremented by 1 from `start`.
        * An iterable of member names.  Values are incremented by 1 from `start`.
        * An iterable of (member name, value) pairs.
        * A mapping of member name -> value pairs.
        """
        metacls = cls.__class__
        bases = (cls,) if type is None else (type, cls)
        _, first_enum = cls._get_mixins_(class_name, bases)
        classdict = metacls.__prepare__(class_name, bases)
        if isinstance(names, str):
            names = names.replace(',', ' ').split()
        if isinstance(names, (tuple, list)) and names and isinstance(names[0], str):
            original_names, names = (names, [])
            last_values = []
            for count, name in enumerate(original_names):
                value = first_enum._generate_next_value_(name, start, count, last_values[:])
                last_values.append(value)
                names.append((name, value))
        if names is None:
            names = ()
        for item in names:
            if isinstance(item, str):
                member_name, member_value = (item, names[item])
            else:
                member_name, member_value = item
            classdict[member_name] = member_value
        if module is None:
            try:
                module = sys._getframe(2).f_globals['__name__']
            except (AttributeError, ValueError, KeyError):
                pass
        if module is None:
            _make_class_unpicklable(classdict)
        else:
            classdict['__module__'] = module
        if qualname is not None:
            classdict['__qualname__'] = qualname
        return metacls.__new__(metacls, class_name, bases, classdict, boundary=boundary)

    def _convert_(cls, name, module, filter, source=None, *, boundary=None, as_global=False):
        """
        Create a new Enum subclass that replaces a collection of global constants
        """
        module_globals = sys.modules[module].__dict__
        if source:
            source = source.__dict__
        else:
            source = module_globals
        members = [(name, value) for name, value in source.items() if filter(name)]
        try:
            members.sort(key=lambda t: (t[1], t[0]))
        except TypeError:
            members.sort(key=lambda t: t[0])
        body = {t[0]: t[1] for t in members}
        body['__module__'] = module
        tmp_cls = type(name, (object,), body)
        cls = _simple_enum(etype=cls, boundary=boundary or KEEP)(tmp_cls)
        if as_global:
            global_enum(cls)
        else:
            sys.modules[cls.__module__].__dict__.update(cls.__members__)
        module_globals[name] = cls
        return cls

    @classmethod
    def _check_for_existing_members_(mcls, class_name, bases):
        for chain in bases:
            for base in chain.__mro__:
                if isinstance(base, EnumType) and base._member_names_:
                    raise TypeError('<enum %r> cannot extend %r' % (class_name, base))

    @classmethod
    def _get_mixins_(mcls, class_name, bases):
        """
        Returns the type for creating enum members, and the first inherited
        enum class.

        bases: the tuple of bases that was given to __new__
        """
        if not bases:
            return (object, Enum)
        mcls._check_for_existing_members_(class_name, bases)
        first_enum = bases[-1]
        if not isinstance(first_enum, EnumType):
            raise TypeError('new enumerations should be created as `EnumName([mixin_type, ...] [data_type,] enum_type)`')
        member_type = mcls._find_data_type_(class_name, bases) or object
        return (member_type, first_enum)

    @classmethod
    def _find_data_repr_(mcls, class_name, bases):
        for chain in bases:
            for base in chain.__mro__:
                if base is object:
                    continue
                elif isinstance(base, EnumType):
                    return base._value_repr_
                elif '__repr__' in base.__dict__:
                    return base.__dict__['__repr__']
        return None

    @classmethod
    def _find_data_type_(mcls, class_name, bases):
        data_types = set()
        base_chain = set()
        for chain in bases:
            candidate = None
            for base in chain.__mro__:
                base_chain.add(base)
                if base is object:
                    continue
                elif isinstance(base, EnumType):
                    if base._member_type_ is not object:
                        data_types.add(base._member_type_)
                        break
                elif '__new__' in base.__dict__ or '__dataclass_fields__' in base.__dict__:
                    if isinstance(base, EnumType):
                        continue
                    data_types.add(candidate or base)
                    break
                else:
                    candidate = candidate or base
        if len(data_types) > 1:
            raise TypeError('too many data types for %r: %r' % (class_name, data_types))
        elif data_types:
            return data_types.pop()
        else:
            return None

    @classmethod
    def _find_new_(mcls, classdict, member_type, first_enum):
        """
        Returns the __new__ to be used for creating the enum members.

        classdict: the class dictionary given to __new__
        member_type: the data type whose __new__ will be used by default
        first_enum: enumeration to check for an overriding __new__
        """
        __new__ = classdict.get('__new__', None)
        save_new = first_enum is not None and __new__ is not None
        if __new__ is None:
            for method in ('__new_member__', '__new__'):
                for possible in (member_type, first_enum):
                    target = getattr(possible, method, None)
                    if target not in {None, None.__new__, object.__new__, Enum.__new__}:
                        __new__ = target
                        break
                if __new__ is not None:
                    break
            else:
                __new__ = object.__new__
        if first_enum is None or __new__ in (Enum.__new__, object.__new__):
            use_args = False
        else:
            use_args = True
        return (__new__, save_new, use_args)