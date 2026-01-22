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
class Documenter:
    """
    A Documenter knows how to autodocument a single object type.  When
    registered with the AutoDirective, it will be used to document objects
    of that type when needed by autodoc.

    Its *objtype* attribute selects what auto directive it is assigned to
    (the directive name is 'auto' + objtype), and what directive it generates
    by default, though that can be overridden by an attribute called
    *directivetype*.

    A Documenter has an *option_spec* that works like a docutils directive's;
    in fact, it will be used to parse an auto directive's options that matches
    the Documenter.
    """
    objtype = 'object'
    content_indent = '   '
    priority = 0
    member_order = 0
    titles_allowed = False
    option_spec: OptionSpec = {'noindex': bool_option}

    def get_attr(self, obj: Any, name: str, *defargs: Any) -> Any:
        """getattr() override for types such as Zope interfaces."""
        return autodoc_attrgetter(self.env.app, obj, name, *defargs)

    @classmethod
    def can_document_member(cls, member: Any, membername: str, isattr: bool, parent: Any) -> bool:
        """Called to see if a member can be documented by this Documenter."""
        raise NotImplementedError('must be implemented in subclasses')

    def __init__(self, directive: 'DocumenterBridge', name: str, indent: str='') -> None:
        self.directive = directive
        self.config: Config = directive.env.config
        self.env: BuildEnvironment = directive.env
        self.options = directive.genopt
        self.name = name
        self.indent = indent
        self.modname: str = None
        self.module: ModuleType = None
        self.objpath: List[str] = None
        self.fullname: str = None
        self.args: str = None
        self.retann: str = None
        self.object: Any = None
        self.object_name: str = None
        self.parent: Any = None
        self.analyzer: ModuleAnalyzer = None

    @property
    def documenters(self) -> Dict[str, Type['Documenter']]:
        """Returns registered Documenter classes"""
        return self.env.app.registry.documenters

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        """Append one line of generated reST to the output."""
        if line.strip():
            self.directive.result.append(self.indent + line, source, *lineno)
        else:
            self.directive.result.append('', source, *lineno)

    def resolve_name(self, modname: str, parents: Any, path: str, base: Any) -> Tuple[str, List[str]]:
        """Resolve the module and name of the object to document given by the
        arguments and the current module/class.

        Must return a pair of the module name and a chain of attributes; for
        example, it would return ``('zipfile', ['ZipFile', 'open'])`` for the
        ``zipfile.ZipFile.open`` method.
        """
        raise NotImplementedError('must be implemented in subclasses')

    def parse_name(self) -> bool:
        """Determine what module to import and what attribute to document.

        Returns True and sets *self.modname*, *self.objpath*, *self.fullname*,
        *self.args* and *self.retann* if parsing and resolving was successful.
        """
        try:
            matched = py_ext_sig_re.match(self.name)
            explicit_modname, path, base, args, retann = matched.groups()
        except AttributeError:
            logger.warning(__('invalid signature for auto%s (%r)') % (self.objtype, self.name), type='autodoc')
            return False
        if explicit_modname is not None:
            modname = explicit_modname[:-2]
            parents = path.rstrip('.').split('.') if path else []
        else:
            modname = None
            parents = []
        with mock(self.config.autodoc_mock_imports):
            self.modname, self.objpath = self.resolve_name(modname, parents, path, base)
        if not self.modname:
            return False
        self.args = args
        self.retann = retann
        self.fullname = (self.modname or '') + ('.' + '.'.join(self.objpath) if self.objpath else '')
        return True

    def import_object(self, raiseerror: bool=False) -> bool:
        """Import the object given by *self.modname* and *self.objpath* and set
        it as *self.object*.

        Returns True if successful, False if an error occurred.
        """
        with mock(self.config.autodoc_mock_imports):
            try:
                ret = import_object(self.modname, self.objpath, self.objtype, attrgetter=self.get_attr, warningiserror=self.config.autodoc_warningiserror)
                self.module, self.parent, self.object_name, self.object = ret
                if ismock(self.object):
                    self.object = undecorate(self.object)
                return True
            except ImportError as exc:
                if raiseerror:
                    raise
                else:
                    logger.warning(exc.args[0], type='autodoc', subtype='import_object')
                    self.env.note_reread()
                    return False

    def get_real_modname(self) -> str:
        """Get the real module name of an object to document.

        It can differ from the name of the module through which the object was
        imported.
        """
        return self.get_attr(self.object, '__module__', None) or self.modname

    def check_module(self) -> bool:
        """Check if *self.object* is really defined in the module given by
        *self.modname*.
        """
        if self.options.imported_members:
            return True
        subject = inspect.unpartial(self.object)
        modname = self.get_attr(subject, '__module__', None)
        if modname and modname != self.modname:
            return False
        return True

    def format_args(self, **kwargs: Any) -> str:
        """Format the argument signature of *self.object*.

        Should return None if the object does not have a signature.
        """
        return None

    def format_name(self) -> str:
        """Format the name of *self.object*.

        This normally should be something that can be parsed by the generated
        directive, but doesn't need to be (Sphinx will display it unparsed
        then).
        """
        return '.'.join(self.objpath) or self.modname

    def _call_format_args(self, **kwargs: Any) -> str:
        if kwargs:
            try:
                return self.format_args(**kwargs)
            except TypeError:
                pass
        return self.format_args()

    def format_signature(self, **kwargs: Any) -> str:
        """Format the signature (arguments and return annotation) of the object.

        Let the user process it via the ``autodoc-process-signature`` event.
        """
        if self.args is not None:
            args = '(%s)' % self.args
            retann = self.retann
        else:
            try:
                retann = None
                args = self._call_format_args(**kwargs)
                if args:
                    matched = re.match('^(\\(.*\\))\\s+->\\s+(.*)$', args)
                    if matched:
                        args = matched.group(1)
                        retann = matched.group(2)
            except Exception as exc:
                logger.warning(__('error while formatting arguments for %s: %s'), self.fullname, exc, type='autodoc')
                args = None
        result = self.env.events.emit_firstresult('autodoc-process-signature', self.objtype, self.fullname, self.object, self.options, args, retann)
        if result:
            args, retann = result
        if args is not None:
            return args + (' -> %s' % retann if retann else '')
        else:
            return ''

    def add_directive_header(self, sig: str) -> None:
        """Add the directive header and options to the generated content."""
        domain = getattr(self, 'domain', 'py')
        directive = getattr(self, 'directivetype', self.objtype)
        name = self.format_name()
        sourcename = self.get_sourcename()
        prefix = '.. %s:%s:: ' % (domain, directive)
        for i, sig_line in enumerate(sig.split('\n')):
            self.add_line('%s%s%s' % (prefix, name, sig_line), sourcename)
            if i == 0:
                prefix = ' ' * len(prefix)
        if self.options.noindex:
            self.add_line('   :noindex:', sourcename)
        if self.objpath:
            self.add_line('   :module: %s' % self.modname, sourcename)

    def get_doc(self) -> Optional[List[List[str]]]:
        """Decode and return lines of the docstring(s) for the object.

        When it returns None, autodoc-process-docstring will not be called for this
        object.
        """
        docstring = getdoc(self.object, self.get_attr, self.config.autodoc_inherit_docstrings, self.parent, self.object_name)
        if docstring:
            tab_width = self.directive.state.document.settings.tab_width
            return [prepare_docstring(docstring, tab_width)]
        return []

    def process_doc(self, docstrings: List[List[str]]) -> Iterator[str]:
        """Let the user process the docstrings before adding them."""
        for docstringlines in docstrings:
            if self.env.app:
                self.env.app.emit('autodoc-process-docstring', self.objtype, self.fullname, self.object, self.options, docstringlines)
                if docstringlines and docstringlines[-1] != '':
                    docstringlines.append('')
            yield from docstringlines

    def get_sourcename(self) -> str:
        if inspect.safe_getattr(self.object, '__module__', None) and inspect.safe_getattr(self.object, '__qualname__', None):
            fullname = '%s.%s' % (self.object.__module__, self.object.__qualname__)
        else:
            fullname = self.fullname
        if self.analyzer:
            return '%s:docstring of %s' % (self.analyzer.srcname, fullname)
        else:
            return 'docstring of %s' % fullname

    def add_content(self, more_content: Optional[StringList]) -> None:
        """Add content from docstrings, attribute documentation and user."""
        docstring = True
        sourcename = self.get_sourcename()
        if self.analyzer:
            attr_docs = self.analyzer.find_attr_docs()
            if self.objpath:
                key = ('.'.join(self.objpath[:-1]), self.objpath[-1])
                if key in attr_docs:
                    docstring = False
                    docstrings = [list(attr_docs[key])]
                    for i, line in enumerate(self.process_doc(docstrings)):
                        self.add_line(line, sourcename, i)
        if docstring:
            docstrings = self.get_doc()
            if docstrings is None:
                pass
            else:
                if not docstrings:
                    docstrings.append([])
                for i, line in enumerate(self.process_doc(docstrings)):
                    self.add_line(line, sourcename, i)
        if more_content:
            for line, src in zip(more_content.data, more_content.items):
                self.add_line(line, src[0], src[1])

    def get_object_members(self, want_all: bool) -> Tuple[bool, ObjectMembers]:
        """Return `(members_check_module, members)` where `members` is a
        list of `(membername, member)` pairs of the members of *self.object*.

        If *want_all* is True, return all members.  Else, only return those
        members given by *self.options.members* (which may also be None).
        """
        warnings.warn('The implementation of Documenter.get_object_members() will be removed from Sphinx-6.0.', RemovedInSphinx60Warning)
        members = get_object_members(self.object, self.objpath, self.get_attr, self.analyzer)
        if not want_all:
            if not self.options.members:
                return (False, [])
            selected = []
            for name in self.options.members:
                if name in members:
                    selected.append((name, members[name].value))
                else:
                    logger.warning(__('missing attribute %s in object %s') % (name, self.fullname), type='autodoc')
            return (False, selected)
        elif self.options.inherited_members:
            return (False, [(m.name, m.value) for m in members.values()])
        else:
            return (False, [(m.name, m.value) for m in members.values() if m.directly_defined])

    def filter_members(self, members: ObjectMembers, want_all: bool) -> List[Tuple[str, Any, bool]]:
        """Filter the given member list.

        Members are skipped if

        - they are private (except if given explicitly or the private-members
          option is set)
        - they are special methods (except if given explicitly or the
          special-members option is set)
        - they are undocumented (except if the undoc-members option is set)

        The user can override the skipping decision by connecting to the
        ``autodoc-skip-member`` event.
        """

        def is_filtered_inherited_member(name: str, obj: Any) -> bool:
            inherited_members = self.options.inherited_members or set()
            if inspect.isclass(self.object):
                for cls in self.object.__mro__:
                    if cls.__name__ in inherited_members and cls != self.object:
                        return True
                    elif name in cls.__dict__:
                        return False
                    elif name in self.get_attr(cls, '__annotations__', {}):
                        return False
                    elif isinstance(obj, ObjectMember) and obj.class_ is cls:
                        return False
            return False
        ret = []
        namespace = '.'.join(self.objpath)
        if self.analyzer:
            attr_docs = self.analyzer.find_attr_docs()
        else:
            attr_docs = {}
        for obj in members:
            try:
                membername, member = obj
                if member is INSTANCEATTR:
                    isattr = True
                elif (namespace, membername) in attr_docs:
                    isattr = True
                else:
                    isattr = False
                doc = getdoc(member, self.get_attr, self.config.autodoc_inherit_docstrings, self.object, membername)
                if not isinstance(doc, str):
                    doc = None
                cls = self.get_attr(member, '__class__', None)
                if cls:
                    cls_doc = self.get_attr(cls, '__doc__', None)
                    if cls_doc == doc:
                        doc = None
                if isinstance(obj, ObjectMember) and obj.docstring:
                    doc = obj.docstring
                doc, metadata = separate_metadata(doc)
                has_doc = bool(doc)
                if 'private' in metadata:
                    isprivate = True
                elif 'public' in metadata:
                    isprivate = False
                else:
                    isprivate = membername.startswith('_')
                keep = False
                if ismock(member) and (namespace, membername) not in attr_docs:
                    pass
                elif self.options.exclude_members and membername in self.options.exclude_members:
                    keep = False
                elif want_all and special_member_re.match(membername):
                    if self.options.special_members and membername in self.options.special_members:
                        if membername == '__doc__':
                            keep = False
                        elif is_filtered_inherited_member(membername, obj):
                            keep = False
                        else:
                            keep = has_doc or self.options.undoc_members
                    else:
                        keep = False
                elif (namespace, membername) in attr_docs:
                    if want_all and isprivate:
                        if self.options.private_members is None:
                            keep = False
                        else:
                            keep = membername in self.options.private_members
                    else:
                        keep = True
                elif want_all and isprivate:
                    if has_doc or self.options.undoc_members:
                        if self.options.private_members is None:
                            keep = False
                        elif is_filtered_inherited_member(membername, obj):
                            keep = False
                        else:
                            keep = membername in self.options.private_members
                    else:
                        keep = False
                elif self.options.members is ALL and is_filtered_inherited_member(membername, obj):
                    keep = False
                else:
                    keep = has_doc or self.options.undoc_members
                if isinstance(obj, ObjectMember) and obj.skipped:
                    keep = False
                if self.env.app:
                    skip_user = self.env.app.emit_firstresult('autodoc-skip-member', self.objtype, membername, member, not keep, self.options)
                    if skip_user is not None:
                        keep = not skip_user
            except Exception as exc:
                logger.warning(__('autodoc: failed to determine %s.%s (%r) to be documented, the following exception was raised:\n%s'), self.name, membername, member, exc, type='autodoc')
                keep = False
            if keep:
                ret.append((membername, member, isattr))
        return ret

    def document_members(self, all_members: bool=False) -> None:
        """Generate reST for member documentation.

        If *all_members* is True, document all members, else those given by
        *self.options.members*.
        """
        self.env.temp_data['autodoc:module'] = self.modname
        if self.objpath:
            self.env.temp_data['autodoc:class'] = self.objpath[0]
        want_all = all_members or self.options.inherited_members or self.options.members is ALL
        members_check_module, members = self.get_object_members(want_all)
        memberdocumenters: List[Tuple[Documenter, bool]] = []
        for mname, member, isattr in self.filter_members(members, want_all):
            classes = [cls for cls in self.documenters.values() if cls.can_document_member(member, mname, isattr, self)]
            if not classes:
                continue
            classes.sort(key=lambda cls: cls.priority)
            full_mname = self.modname + '::' + '.'.join(self.objpath + [mname])
            documenter = classes[-1](self.directive, full_mname, self.indent)
            memberdocumenters.append((documenter, isattr))
        member_order = self.options.member_order or self.config.autodoc_member_order
        memberdocumenters = self.sort_members(memberdocumenters, member_order)
        for documenter, isattr in memberdocumenters:
            documenter.generate(all_members=True, real_modname=self.real_modname, check_module=members_check_module and (not isattr))
        self.env.temp_data['autodoc:module'] = None
        self.env.temp_data['autodoc:class'] = None

    def sort_members(self, documenters: List[Tuple['Documenter', bool]], order: str) -> List[Tuple['Documenter', bool]]:
        """Sort the given member list."""
        if order == 'groupwise':
            documenters.sort(key=lambda e: (e[0].member_order, e[0].name))
        elif order == 'bysource':
            if self.analyzer:
                tagorder = self.analyzer.tagorder

                def keyfunc(entry: Tuple[Documenter, bool]) -> int:
                    fullname = entry[0].name.split('::')[1]
                    return tagorder.get(fullname, len(tagorder))
                documenters.sort(key=keyfunc)
            else:
                pass
        else:
            documenters.sort(key=lambda e: e[0].name)
        return documenters

    def generate(self, more_content: Optional[StringList]=None, real_modname: str=None, check_module: bool=False, all_members: bool=False) -> None:
        """Generate reST for the object given by *self.name*, and possibly for
        its members.

        If *more_content* is given, include that content. If *real_modname* is
        given, use that module name to find attribute docs. If *check_module* is
        True, only generate if the object is defined in the module name it is
        imported from. If *all_members* is True, document all members.
        """
        if not self.parse_name():
            logger.warning(__('don\'t know which module to import for autodocumenting %r (try placing a "module" or "currentmodule" directive in the document, or giving an explicit module name)') % self.name, type='autodoc')
            return
        if not self.import_object():
            return
        guess_modname = self.get_real_modname()
        self.real_modname: str = real_modname or guess_modname
        try:
            self.analyzer = ModuleAnalyzer.for_module(self.real_modname)
            self.analyzer.find_attr_docs()
        except PycodeError as exc:
            logger.debug('[autodoc] module analyzer failed: %s', exc)
            self.analyzer = None
            if hasattr(self.module, '__file__') and self.module.__file__:
                self.directive.record_dependencies.add(self.module.__file__)
        else:
            self.directive.record_dependencies.add(self.analyzer.srcname)
        if self.real_modname != guess_modname:
            try:
                analyzer = ModuleAnalyzer.for_module(guess_modname)
                self.directive.record_dependencies.add(analyzer.srcname)
            except PycodeError:
                pass
        docstrings: List[str] = sum(self.get_doc() or [], [])
        if ismock(self.object) and (not docstrings):
            logger.warning(__('A mocked object is detected: %r'), self.name, type='autodoc')
        if check_module:
            if not self.check_module():
                return
        sourcename = self.get_sourcename()
        self.add_line('', sourcename)
        try:
            sig = self.format_signature()
        except Exception as exc:
            logger.warning(__('error while formatting signature for %s: %s'), self.fullname, exc, type='autodoc')
            return
        self.add_directive_header(sig)
        self.add_line('', sourcename)
        self.indent += self.content_indent
        self.add_content(more_content)
        self.document_members(all_members)