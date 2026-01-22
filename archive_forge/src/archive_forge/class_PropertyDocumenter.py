import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence,
from docutils.statemachine import StringList
import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import (get_class_members, get_object_members, import_module,
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
from sphinx.util.typing import OptionSpec, get_type_hints, restify
from sphinx.util.typing import stringify as stringify_typehint
class PropertyDocumenter(DocstringStripSignatureMixin, ClassLevelDocumenter):
    """
    Specialized Documenter subclass for properties.
    """
    objtype = 'property'
    member_order = 60
    priority = AttributeDocumenter.priority + 1

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        if isinstance(parent, ClassDocumenter):
            if inspect.isproperty(member):
                return True
            else:
                __dict__ = safe_getattr(parent.object, '__dict__', {})
                obj = __dict__.get(membername)
                return isinstance(obj, classmethod) and inspect.isproperty(obj.__func__)
        else:
            return False

    def import_object(self, raiseerror: bool=False) -> bool:
        """Check the exisitence of uninitialized instance attribute when failed to import
        the attribute."""
        ret = super().import_object(raiseerror)
        if ret and (not inspect.isproperty(self.object)):
            __dict__ = safe_getattr(self.parent, '__dict__', {})
            obj = __dict__.get(self.objpath[-1])
            if isinstance(obj, classmethod) and inspect.isproperty(obj.__func__):
                self.object = obj.__func__
                self.isclassmethod = True
                return True
            else:
                return False
        self.isclassmethod = False
        return ret

    def document_members(self, all_members: bool=False) -> None:
        pass

    def get_real_modname(self) -> str:
        real_modname = self.get_attr(self.parent or self.object, '__module__', None)
        return real_modname or self.modname

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if inspect.isabstractmethod(self.object):
            self.add_line('   :abstractmethod:', sourcename)
        if self.isclassmethod:
            self.add_line('   :classmethod:', sourcename)
        if safe_getattr(self.object, 'fget', None):
            func = self.object.fget
        elif safe_getattr(self.object, 'func', None):
            func = self.object.func
        else:
            func = None
        if func and self.config.autodoc_typehints != 'none':
            try:
                signature = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
                if signature.return_annotation is not Parameter.empty:
                    if self.config.autodoc_typehints_format == 'short':
                        objrepr = stringify_typehint(signature.return_annotation, 'smart')
                    else:
                        objrepr = stringify_typehint(signature.return_annotation)
                    self.add_line('   :type: ' + objrepr, sourcename)
            except TypeError as exc:
                logger.warning(__('Failed to get a function signature for %s: %s'), self.fullname, exc)
                return None
            except ValueError:
                return None