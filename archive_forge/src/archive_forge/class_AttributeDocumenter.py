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
class AttributeDocumenter(GenericAliasMixin, NewTypeMixin, SlotsMixin, TypeVarMixin, RuntimeInstanceAttributeMixin, UninitializedInstanceAttributeMixin, NonDataDescriptorMixin, DocstringStripSignatureMixin, ClassLevelDocumenter):
    """
    Specialized Documenter subclass for attributes.
    """
    objtype = 'attribute'
    member_order = 60
    option_spec: OptionSpec = dict(ModuleLevelDocumenter.option_spec)
    option_spec['annotation'] = annotation_option
    option_spec['no-value'] = bool_option
    priority = 10

    @staticmethod
    def is_function_or_method(obj: Any) -> bool:
        return inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        if isinstance(parent, ModuleDocumenter):
            return False
        elif inspect.isattributedescriptor(member):
            return True
        elif not inspect.isroutine(member) and (not isinstance(member, type)):
            return True
        else:
            return False

    def document_members(self, all_members: bool=False) -> None:
        pass

    def update_annotations(self, parent: Any) -> None:
        """Update __annotations__ to support type_comment and so on."""
        try:
            annotations = dict(inspect.getannotations(parent))
            parent.__annotations__ = annotations
            for cls in inspect.getmro(parent):
                try:
                    module = safe_getattr(cls, '__module__')
                    qualname = safe_getattr(cls, '__qualname__')
                    analyzer = ModuleAnalyzer.for_module(module)
                    analyzer.analyze()
                    for (classname, attrname), annotation in analyzer.annotations.items():
                        if classname == qualname and attrname not in annotations:
                            annotations[attrname] = annotation
                except (AttributeError, PycodeError):
                    pass
        except (AttributeError, TypeError):
            pass

    def import_object(self, raiseerror: bool=False) -> bool:
        ret = super().import_object(raiseerror)
        if inspect.isenumattribute(self.object):
            self.object = self.object.value
        if self.parent:
            self.update_annotations(self.parent)
        return ret

    def get_real_modname(self) -> str:
        real_modname = self.get_attr(self.parent or self.object, '__module__', None)
        return real_modname or self.modname

    def should_suppress_value_header(self) -> bool:
        if super().should_suppress_value_header():
            return True
        else:
            doc = self.get_doc()
            if doc:
                docstring, metadata = separate_metadata('\n'.join(sum(doc, [])))
                if 'hide-value' in metadata:
                    return True
        return False

    def add_directive_header(self, sig: str) -> None:
        super().add_directive_header(sig)
        sourcename = self.get_sourcename()
        if self.options.annotation is SUPPRESS or self.should_suppress_directive_header():
            pass
        elif self.options.annotation:
            self.add_line('   :annotation: %s' % self.options.annotation, sourcename)
        else:
            if self.config.autodoc_typehints != 'none':
                annotations = get_type_hints(self.parent, None, self.config.autodoc_type_aliases)
                if self.objpath[-1] in annotations:
                    if self.config.autodoc_typehints_format == 'short':
                        objrepr = stringify_typehint(annotations.get(self.objpath[-1]), 'smart')
                    else:
                        objrepr = stringify_typehint(annotations.get(self.objpath[-1]))
                    self.add_line('   :type: ' + objrepr, sourcename)
            try:
                if self.options.no_value or self.should_suppress_value_header() or ismock(self.object):
                    pass
                else:
                    objrepr = object_description(self.object)
                    self.add_line('   :value: ' + objrepr, sourcename)
            except ValueError:
                pass

    def get_attribute_comment(self, parent: Any, attrname: str) -> Optional[List[str]]:
        for cls in inspect.getmro(parent):
            try:
                module = safe_getattr(cls, '__module__')
                qualname = safe_getattr(cls, '__qualname__')
                analyzer = ModuleAnalyzer.for_module(module)
                analyzer.analyze()
                if qualname and self.objpath:
                    key = (qualname, attrname)
                    if key in analyzer.attr_docs:
                        return list(analyzer.attr_docs[key])
            except (AttributeError, PycodeError):
                pass
        return None

    def get_doc(self) -> Optional[List[List[str]]]:
        comment = self.get_attribute_comment(self.parent, self.objpath[-1])
        if comment:
            return [comment]
        try:
            orig = self.config.autodoc_inherit_docstrings
            self.config.autodoc_inherit_docstrings = False
            return super().get_doc()
        finally:
            self.config.autodoc_inherit_docstrings = orig

    def add_content(self, more_content: Optional[StringList]) -> None:
        self.analyzer = None
        if more_content is None:
            more_content = StringList()
        self.update_content(more_content)
        super().add_content(more_content)