import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
class MagicsManager(Configurable):
    """Object that handles all magic-related functionality for IPython.
    """
    magics = Dict()
    lazy_magics = Dict(help='\n    Mapping from magic names to modules to load.\n\n    This can be used in IPython/IPykernel configuration to declare lazy magics\n    that will only be imported/registered on first use.\n\n    For example::\n\n        c.MagicsManager.lazy_magics = {\n          "my_magic": "slow.to.import",\n          "my_other_magic": "also.slow",\n        }\n\n    On first invocation of `%my_magic`, `%%my_magic`, `%%my_other_magic` or\n    `%%my_other_magic`, the corresponding module will be loaded as an ipython\n    extensions as if you had previously done `%load_ext ipython`.\n\n    Magics names should be without percent(s) as magics can be both cell\n    and line magics.\n\n    Lazy loading happen relatively late in execution process, and\n    complex extensions that manipulate Python/IPython internal state or global state\n    might not support lazy loading.\n    ').tag(config=True)
    registry = Dict()
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    auto_magic = Bool(True, help='Automatically call line magics without requiring explicit % prefix').tag(config=True)

    @observe('auto_magic')
    def _auto_magic_changed(self, change):
        self.shell.automagic = change['new']
    _auto_status = ['Automagic is OFF, % prefix IS needed for line magics.', 'Automagic is ON, % prefix IS NOT needed for line magics.']
    user_magics = Instance('IPython.core.magics.UserMagics', allow_none=True)

    def __init__(self, shell=None, config=None, user_magics=None, **traits):
        super(MagicsManager, self).__init__(shell=shell, config=config, user_magics=user_magics, **traits)
        self.magics = dict(line={}, cell={})
        self.registry[user_magics.__class__.__name__] = user_magics

    def auto_status(self):
        """Return descriptive string with automagic status."""
        return self._auto_status[self.auto_magic]

    def lsmagic(self):
        """Return a dict of currently available magic functions.

        The return dict has the keys 'line' and 'cell', corresponding to the
        two types of magics we support.  Each value is a list of names.
        """
        return self.magics

    def lsmagic_docs(self, brief=False, missing=''):
        """Return dict of documentation of magic functions.

        The return dict has the keys 'line' and 'cell', corresponding to the
        two types of magics we support. Each value is a dict keyed by magic
        name whose value is the function docstring. If a docstring is
        unavailable, the value of `missing` is used instead.

        If brief is True, only the first line of each docstring will be returned.
        """
        docs = {}
        for m_type in self.magics:
            m_docs = {}
            for m_name, m_func in self.magics[m_type].items():
                if m_func.__doc__:
                    if brief:
                        m_docs[m_name] = m_func.__doc__.split('\n', 1)[0]
                    else:
                        m_docs[m_name] = m_func.__doc__.rstrip()
                else:
                    m_docs[m_name] = missing
            docs[m_type] = m_docs
        return docs

    def register_lazy(self, name: str, fully_qualified_name: str):
        """
        Lazily register a magic via an extension.


        Parameters
        ----------
        name : str
            Name of the magic you wish to register.
        fully_qualified_name :
            Fully qualified name of the module/submodule that should be loaded
            as an extensions when the magic is first called.
            It is assumed that loading this extensions will register the given
            magic.
        """
        self.lazy_magics[name] = fully_qualified_name

    def register(self, *magic_objects):
        """Register one or more instances of Magics.

        Take one or more classes or instances of classes that subclass the main
        `core.Magic` class, and register them with IPython to use the magic
        functions they provide.  The registration process will then ensure that
        any methods that have decorated to provide line and/or cell magics will
        be recognized with the `%x`/`%%x` syntax as a line/cell magic
        respectively.

        If classes are given, they will be instantiated with the default
        constructor.  If your classes need a custom constructor, you should
        instanitate them first and pass the instance.

        The provided arguments can be an arbitrary mix of classes and instances.

        Parameters
        ----------
        *magic_objects : one or more classes or instances
        """
        for m in magic_objects:
            if not m.registered:
                raise ValueError('Class of magics %r was constructed without the @register_magics class decorator')
            if isinstance(m, type):
                m = m(shell=self.shell)
            self.registry[m.__class__.__name__] = m
            for mtype in magic_kinds:
                self.magics[mtype].update(m.magics[mtype])

    def register_function(self, func, magic_kind='line', magic_name=None):
        """Expose a standalone function as magic function for IPython.

        This will create an IPython magic (line, cell or both) from a
        standalone function.  The functions should have the following
        signatures:

        * For line magics: `def f(line)`
        * For cell magics: `def f(line, cell)`
        * For a function that does both: `def f(line, cell=None)`

        In the latter case, the function will be called with `cell==None` when
        invoked as `%f`, and with cell as a string when invoked as `%%f`.

        Parameters
        ----------
        func : callable
            Function to be registered as a magic.
        magic_kind : str
            Kind of magic, one of 'line', 'cell' or 'line_cell'
        magic_name : optional str
            If given, the name the magic will have in the IPython namespace.  By
            default, the name of the function itself is used.
        """
        validate_type(magic_kind)
        magic_name = func.__name__ if magic_name is None else magic_name
        setattr(self.user_magics, magic_name, func)
        record_magic(self.magics, magic_kind, magic_name, func)

    def register_alias(self, alias_name, magic_name, magic_kind='line', magic_params=None):
        """Register an alias to a magic function.

        The alias is an instance of :class:`MagicAlias`, which holds the
        name and kind of the magic it should call. Binding is done at
        call time, so if the underlying magic function is changed the alias
        will call the new function.

        Parameters
        ----------
        alias_name : str
            The name of the magic to be registered.
        magic_name : str
            The name of an existing magic.
        magic_kind : str
            Kind of magic, one of 'line' or 'cell'
        """
        if magic_kind not in magic_kinds:
            raise ValueError('magic_kind must be one of %s, %s given' % magic_kinds, magic_kind)
        alias = MagicAlias(self.shell, magic_name, magic_kind, magic_params)
        setattr(self.user_magics, alias_name, alias)
        record_magic(self.magics, magic_kind, alias_name, alias)