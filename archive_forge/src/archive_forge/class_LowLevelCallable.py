from . import _ccallback_c
import ctypes
class LowLevelCallable(tuple):
    """
    Low-level callback function.

    Some functions in SciPy take as arguments callback functions, which
    can either be python callables or low-level compiled functions. Using
    compiled callback functions can improve performance somewhat by
    avoiding wrapping data in Python objects.

    Such low-level functions in SciPy are wrapped in `LowLevelCallable`
    objects, which can be constructed from function pointers obtained from
    ctypes, cffi, Cython, or contained in Python `PyCapsule` objects.

    .. seealso::

       Functions accepting low-level callables:

       `scipy.integrate.quad`, `scipy.ndimage.generic_filter`,
       `scipy.ndimage.generic_filter1d`, `scipy.ndimage.geometric_transform`

       Usage examples:

       :ref:`ndimage-ccallbacks`, :ref:`quad-callbacks`

    Parameters
    ----------
    function : {PyCapsule, ctypes function pointer, cffi function pointer}
        Low-level callback function.
    user_data : {PyCapsule, ctypes void pointer, cffi void pointer}
        User data to pass on to the callback function.
    signature : str, optional
        Signature of the function. If omitted, determined from *function*,
        if possible.

    Attributes
    ----------
    function
        Callback function given.
    user_data
        User data given.
    signature
        Signature of the function.

    Methods
    -------
    from_cython
        Class method for constructing callables from Cython C-exported
        functions.

    Notes
    -----
    The argument ``function`` can be one of:

    - PyCapsule, whose name contains the C function signature
    - ctypes function pointer
    - cffi function pointer

    The signature of the low-level callback must match one of those expected
    by the routine it is passed to.

    If constructing low-level functions from a PyCapsule, the name of the
    capsule must be the corresponding signature, in the format::

        return_type (arg1_type, arg2_type, ...)

    For example::

        "void (double)"
        "double (double, int *, void *)"

    The context of a PyCapsule passed in as ``function`` is used as ``user_data``,
    if an explicit value for ``user_data`` was not given.

    """
    __slots__ = ()

    def __new__(cls, function, user_data=None, signature=None):
        item = cls._parse_callback(function, user_data, signature)
        return tuple.__new__(cls, (item, function, user_data))

    def __repr__(self):
        return f'LowLevelCallable({self.function!r}, {self.user_data!r})'

    @property
    def function(self):
        return tuple.__getitem__(self, 1)

    @property
    def user_data(self):
        return tuple.__getitem__(self, 2)

    @property
    def signature(self):
        return _ccallback_c.get_capsule_signature(tuple.__getitem__(self, 0))

    def __getitem__(self, idx):
        raise ValueError()

    @classmethod
    def from_cython(cls, module, name, user_data=None, signature=None):
        """
        Create a low-level callback function from an exported Cython function.

        Parameters
        ----------
        module : module
            Cython module where the exported function resides
        name : str
            Name of the exported function
        user_data : {PyCapsule, ctypes void pointer, cffi void pointer}, optional
            User data to pass on to the callback function.
        signature : str, optional
            Signature of the function. If omitted, determined from *function*.

        """
        try:
            function = module.__pyx_capi__[name]
        except AttributeError as e:
            message = 'Given module is not a Cython module with __pyx_capi__ attribute'
            raise ValueError(message) from e
        except KeyError as e:
            message = f'No function {name!r} found in __pyx_capi__ of the module'
            raise ValueError(message) from e
        return cls(function, user_data, signature)

    @classmethod
    def _parse_callback(cls, obj, user_data=None, signature=None):
        _import_cffi()
        if isinstance(obj, LowLevelCallable):
            func = tuple.__getitem__(obj, 0)
        elif isinstance(obj, PyCFuncPtr):
            func, signature = _get_ctypes_func(obj, signature)
        elif isinstance(obj, CData):
            func, signature = _get_cffi_func(obj, signature)
        elif _ccallback_c.check_capsule(obj):
            func = obj
        else:
            raise ValueError('Given input is not a callable or a low-level callable (pycapsule/ctypes/cffi)')
        if isinstance(user_data, ctypes.c_void_p):
            context = _get_ctypes_data(user_data)
        elif isinstance(user_data, CData):
            context = _get_cffi_data(user_data)
        elif user_data is None:
            context = 0
        elif _ccallback_c.check_capsule(user_data):
            context = user_data
        else:
            raise ValueError('Given user data is not a valid low-level void* pointer (pycapsule/ctypes/cffi)')
        return _ccallback_c.get_raw_capsule(func, signature, context)