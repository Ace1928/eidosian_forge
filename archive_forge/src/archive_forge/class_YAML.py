from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import warnings
import glob
from importlib import import_module
import ruamel.yaml
from ruamel.yaml.error import UnsafeLoaderWarning, YAMLError  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.loader import BaseLoader, SafeLoader, Loader, RoundTripLoader  # NOQA
from ruamel.yaml.dumper import BaseDumper, SafeDumper, Dumper, RoundTripDumper  # NOQA
from ruamel.yaml.compat import StringIO, BytesIO, with_metaclass, PY3, nprint
from ruamel.yaml.resolver import VersionedResolver, Resolver  # NOQA
from ruamel.yaml.representer import (
from ruamel.yaml.constructor import (
from ruamel.yaml.loader import Loader as UnsafeLoader
class YAML(object):

    def __init__(self, _kw=enforce, typ=None, pure=False, output=None, plug_ins=None):
        """
        _kw: not used, forces keyword arguments in 2.7 (in 3 you can do (*, safe_load=..)
        typ: 'rt'/None -> RoundTripLoader/RoundTripDumper,  (default)
             'safe'    -> SafeLoader/SafeDumper,
             'unsafe'  -> normal/unsafe Loader/Dumper
             'base'    -> baseloader
        pure: if True only use Python modules
        input/output: needed to work as context manager
        plug_ins: a list of plug-in files
        """
        if _kw is not enforce:
            raise TypeError('{}.__init__() takes no positional argument but at least one was given ({!r})'.format(self.__class__.__name__, _kw))
        self.typ = 'rt' if typ is None else typ
        self.pure = pure
        self._output = output
        self._context_manager = None
        self.plug_ins = []
        for pu in ([] if plug_ins is None else plug_ins) + self.official_plug_ins():
            file_name = pu.replace(os.sep, '.')
            self.plug_ins.append(import_module(file_name))
        self.Resolver = ruamel.yaml.resolver.VersionedResolver
        self.allow_unicode = True
        self.Reader = None
        self.Scanner = None
        self.Serializer = None
        self.default_flow_style = None
        if self.typ == 'rt':
            self.default_flow_style = False
            self.Emitter = ruamel.yaml.emitter.Emitter
            self.Serializer = ruamel.yaml.serializer.Serializer
            self.Representer = ruamel.yaml.representer.RoundTripRepresenter
            self.Scanner = ruamel.yaml.scanner.RoundTripScanner
            self.Parser = ruamel.yaml.parser.RoundTripParser
            self.Composer = ruamel.yaml.composer.Composer
            self.Constructor = ruamel.yaml.constructor.RoundTripConstructor
        elif self.typ == 'safe':
            self.Emitter = ruamel.yaml.emitter.Emitter if pure or CEmitter is None else CEmitter
            self.Representer = ruamel.yaml.representer.SafeRepresenter
            self.Parser = ruamel.yaml.parser.Parser if pure or CParser is None else CParser
            self.Composer = ruamel.yaml.composer.Composer
            self.Constructor = ruamel.yaml.constructor.SafeConstructor
        elif self.typ == 'base':
            self.Emitter = ruamel.yaml.emitter.Emitter
            self.Representer = ruamel.yaml.representer.BaseRepresenter
            self.Parser = ruamel.yaml.parser.Parser if pure or CParser is None else CParser
            self.Composer = ruamel.yaml.composer.Composer
            self.Constructor = ruamel.yaml.constructor.BaseConstructor
        elif self.typ == 'unsafe':
            self.Emitter = ruamel.yaml.emitter.Emitter if pure or CEmitter is None else CEmitter
            self.Representer = ruamel.yaml.representer.Representer
            self.Parser = ruamel.yaml.parser.Parser if pure or CParser is None else CParser
            self.Composer = ruamel.yaml.composer.Composer
            self.Constructor = ruamel.yaml.constructor.Constructor
        else:
            for module in self.plug_ins:
                if getattr(module, 'typ', None) == self.typ:
                    module.init_typ(self)
                    break
            else:
                raise NotImplementedError('typ "{}"not recognised (need to install plug-in?)'.format(self.typ))
        self.stream = None
        self.canonical = None
        self.old_indent = None
        self.width = None
        self.line_break = None
        self.map_indent = None
        self.sequence_indent = None
        self.sequence_dash_offset = 0
        self.compact_seq_seq = None
        self.compact_seq_map = None
        self.sort_base_mapping_type_on_output = None
        self.top_level_colon_align = None
        self.prefix_colon = None
        self.version = None
        self.preserve_quotes = None
        self.allow_duplicate_keys = False
        self.encoding = 'utf-8'
        self.explicit_start = None
        self.explicit_end = None
        self.tags = None
        self.default_style = None
        self.top_level_block_style_scalar_no_indent_error_1_1 = False
        self.brace_single_entry_mapping_in_flow_sequence = False

    @property
    def reader(self):
        try:
            return self._reader
        except AttributeError:
            self._reader = self.Reader(None, loader=self)
            return self._reader

    @property
    def scanner(self):
        try:
            return self._scanner
        except AttributeError:
            self._scanner = self.Scanner(loader=self)
            return self._scanner

    @property
    def parser(self):
        attr = '_' + sys._getframe().f_code.co_name
        if not hasattr(self, attr):
            if self.Parser is not CParser:
                setattr(self, attr, self.Parser(loader=self))
            elif getattr(self, '_stream', None) is None:
                return None
            else:
                setattr(self, attr, CParser(self._stream))
        return getattr(self, attr)

    @property
    def composer(self):
        attr = '_' + sys._getframe().f_code.co_name
        if not hasattr(self, attr):
            setattr(self, attr, self.Composer(loader=self))
        return getattr(self, attr)

    @property
    def constructor(self):
        attr = '_' + sys._getframe().f_code.co_name
        if not hasattr(self, attr):
            cnst = self.Constructor(preserve_quotes=self.preserve_quotes, loader=self)
            cnst.allow_duplicate_keys = self.allow_duplicate_keys
            setattr(self, attr, cnst)
        return getattr(self, attr)

    @property
    def resolver(self):
        attr = '_' + sys._getframe().f_code.co_name
        if not hasattr(self, attr):
            setattr(self, attr, self.Resolver(version=self.version, loader=self))
        return getattr(self, attr)

    @property
    def emitter(self):
        attr = '_' + sys._getframe().f_code.co_name
        if not hasattr(self, attr):
            if self.Emitter is not CEmitter:
                _emitter = self.Emitter(None, canonical=self.canonical, indent=self.old_indent, width=self.width, allow_unicode=self.allow_unicode, line_break=self.line_break, prefix_colon=self.prefix_colon, brace_single_entry_mapping_in_flow_sequence=self.brace_single_entry_mapping_in_flow_sequence, dumper=self)
                setattr(self, attr, _emitter)
                if self.map_indent is not None:
                    _emitter.best_map_indent = self.map_indent
                if self.sequence_indent is not None:
                    _emitter.best_sequence_indent = self.sequence_indent
                if self.sequence_dash_offset is not None:
                    _emitter.sequence_dash_offset = self.sequence_dash_offset
                if self.compact_seq_seq is not None:
                    _emitter.compact_seq_seq = self.compact_seq_seq
                if self.compact_seq_map is not None:
                    _emitter.compact_seq_map = self.compact_seq_map
            else:
                if getattr(self, '_stream', None) is None:
                    return None
                return None
        return getattr(self, attr)

    @property
    def serializer(self):
        attr = '_' + sys._getframe().f_code.co_name
        if not hasattr(self, attr):
            setattr(self, attr, self.Serializer(encoding=self.encoding, explicit_start=self.explicit_start, explicit_end=self.explicit_end, version=self.version, tags=self.tags, dumper=self))
        return getattr(self, attr)

    @property
    def representer(self):
        attr = '_' + sys._getframe().f_code.co_name
        if not hasattr(self, attr):
            repres = self.Representer(default_style=self.default_style, default_flow_style=self.default_flow_style, dumper=self)
            if self.sort_base_mapping_type_on_output is not None:
                repres.sort_base_mapping_type_on_output = self.sort_base_mapping_type_on_output
            setattr(self, attr, repres)
        return getattr(self, attr)

    def load(self, stream):
        """
        at this point you either have the non-pure Parser (which has its own reader and
        scanner) or you have the pure Parser.
        If the pure Parser is set, then set the Reader and Scanner, if not already set.
        If either the Scanner or Reader are set, you cannot use the non-pure Parser,
            so reset it to the pure parser and set the Reader resp. Scanner if necessary
        """
        if not hasattr(stream, 'read') and hasattr(stream, 'open'):
            with stream.open('rb') as fp:
                return self.load(fp)
        constructor, parser = self.get_constructor_parser(stream)
        try:
            return constructor.get_single_data()
        finally:
            parser.dispose()
            try:
                self._reader.reset_reader()
            except AttributeError:
                pass
            try:
                self._scanner.reset_scanner()
            except AttributeError:
                pass

    def load_all(self, stream, _kw=enforce):
        if _kw is not enforce:
            raise TypeError('{}.__init__() takes no positional argument but at least one was given ({!r})'.format(self.__class__.__name__, _kw))
        if not hasattr(stream, 'read') and hasattr(stream, 'open'):
            with stream.open('r') as fp:
                for d in self.load_all(fp, _kw=enforce):
                    yield d
                return
        constructor, parser = self.get_constructor_parser(stream)
        try:
            while constructor.check_data():
                yield constructor.get_data()
        finally:
            parser.dispose()
            try:
                self._reader.reset_reader()
            except AttributeError:
                pass
            try:
                self._scanner.reset_scanner()
            except AttributeError:
                pass

    def get_constructor_parser(self, stream):
        """
        the old cyaml needs special setup, and therefore the stream
        """
        if self.Parser is not CParser:
            if self.Reader is None:
                self.Reader = ruamel.yaml.reader.Reader
            if self.Scanner is None:
                self.Scanner = ruamel.yaml.scanner.Scanner
            self.reader.stream = stream
        elif self.Reader is not None:
            if self.Scanner is None:
                self.Scanner = ruamel.yaml.scanner.Scanner
            self.Parser = ruamel.yaml.parser.Parser
            self.reader.stream = stream
        elif self.Scanner is not None:
            if self.Reader is None:
                self.Reader = ruamel.yaml.reader.Reader
            self.Parser = ruamel.yaml.parser.Parser
            self.reader.stream = stream
        else:
            rslvr = self.Resolver

            class XLoader(self.Parser, self.Constructor, rslvr):

                def __init__(selfx, stream, version=self.version, preserve_quotes=None):
                    CParser.__init__(selfx, stream)
                    selfx._parser = selfx._composer = selfx
                    self.Constructor.__init__(selfx, loader=selfx)
                    selfx.allow_duplicate_keys = self.allow_duplicate_keys
                    rslvr.__init__(selfx, version=version, loadumper=selfx)
            self._stream = stream
            loader = XLoader(stream)
            return (loader, loader)
        return (self.constructor, self.parser)

    def dump(self, data, stream=None, _kw=enforce, transform=None):
        if self._context_manager:
            if not self._output:
                raise TypeError('Missing output stream while dumping from context manager')
            if _kw is not enforce:
                raise TypeError('{}.dump() takes one positional argument but at least two were given ({!r})'.format(self.__class__.__name__, _kw))
            if transform is not None:
                raise TypeError('{}.dump() in the context manager cannot have transform keyword '.format(self.__class__.__name__))
            self._context_manager.dump(data)
        else:
            if stream is None:
                raise TypeError('Need a stream argument when not dumping from context manager')
            return self.dump_all([data], stream, _kw, transform=transform)

    def dump_all(self, documents, stream, _kw=enforce, transform=None):
        if self._context_manager:
            raise NotImplementedError
        if _kw is not enforce:
            raise TypeError('{}.dump(_all) takes two positional argument but at least three were given ({!r})'.format(self.__class__.__name__, _kw))
        self._output = stream
        self._context_manager = YAMLContextManager(self, transform=transform)
        for data in documents:
            self._context_manager.dump(data)
        self._context_manager.teardown_output()
        self._output = None
        self._context_manager = None

    def Xdump_all(self, documents, stream, _kw=enforce, transform=None):
        """
        Serialize a sequence of Python objects into a YAML stream.
        """
        if not hasattr(stream, 'write') and hasattr(stream, 'open'):
            with stream.open('w') as fp:
                return self.dump_all(documents, fp, _kw, transform=transform)
        if _kw is not enforce:
            raise TypeError('{}.dump(_all) takes two positional argument but at least three were given ({!r})'.format(self.__class__.__name__, _kw))
        if self.top_level_colon_align is True:
            tlca = max([len(str(x)) for x in documents[0]])
        else:
            tlca = self.top_level_colon_align
        if transform is not None:
            fstream = stream
            if self.encoding is None:
                stream = StringIO()
            else:
                stream = BytesIO()
        serializer, representer, emitter = self.get_serializer_representer_emitter(stream, tlca)
        try:
            self.serializer.open()
            for data in documents:
                try:
                    self.representer.represent(data)
                except AttributeError:
                    raise
            self.serializer.close()
        finally:
            try:
                self.emitter.dispose()
            except AttributeError:
                raise
            delattr(self, '_serializer')
            delattr(self, '_emitter')
        if transform:
            val = stream.getvalue()
            if self.encoding:
                val = val.decode(self.encoding)
            if fstream is None:
                transform(val)
            else:
                fstream.write(transform(val))
        return None

    def get_serializer_representer_emitter(self, stream, tlca):
        if self.Emitter is not CEmitter:
            if self.Serializer is None:
                self.Serializer = ruamel.yaml.serializer.Serializer
            self.emitter.stream = stream
            self.emitter.top_level_colon_align = tlca
            return (self.serializer, self.representer, self.emitter)
        if self.Serializer is not None:
            self.Emitter = ruamel.yaml.emitter.Emitter
            self.emitter.stream = stream
            self.emitter.top_level_colon_align = tlca
            return (self.serializer, self.representer, self.emitter)
        rslvr = ruamel.yaml.resolver.BaseResolver if self.typ == 'base' else ruamel.yaml.resolver.Resolver

        class XDumper(CEmitter, self.Representer, rslvr):

            def __init__(selfx, stream, default_style=None, default_flow_style=None, canonical=None, indent=None, width=None, allow_unicode=None, line_break=None, encoding=None, explicit_start=None, explicit_end=None, version=None, tags=None, block_seq_indent=None, top_level_colon_align=None, prefix_colon=None):
                CEmitter.__init__(selfx, stream, canonical=canonical, indent=indent, width=width, encoding=encoding, allow_unicode=allow_unicode, line_break=line_break, explicit_start=explicit_start, explicit_end=explicit_end, version=version, tags=tags)
                selfx._emitter = selfx._serializer = selfx._representer = selfx
                self.Representer.__init__(selfx, default_style=default_style, default_flow_style=default_flow_style)
                rslvr.__init__(selfx)
        self._stream = stream
        dumper = XDumper(stream, default_style=self.default_style, default_flow_style=self.default_flow_style, canonical=self.canonical, indent=self.old_indent, width=self.width, allow_unicode=self.allow_unicode, line_break=self.line_break, explicit_start=self.explicit_start, explicit_end=self.explicit_end, version=self.version, tags=self.tags)
        self._emitter = self._serializer = dumper
        return (dumper, dumper, dumper)

    def map(self, **kw):
        if self.typ == 'rt':
            from ruamel.yaml.comments import CommentedMap
            return CommentedMap(**kw)
        else:
            return dict(**kw)

    def seq(self, *args):
        if self.typ == 'rt':
            from ruamel.yaml.comments import CommentedSeq
            return CommentedSeq(*args)
        else:
            return list(*args)

    def official_plug_ins(self):
        bd = os.path.dirname(__file__)
        gpbd = os.path.dirname(os.path.dirname(bd))
        res = [x.replace(gpbd, '')[1:-3] for x in glob.glob(bd + '/*/__plug_in__.py')]
        return res

    def register_class(self, cls):
        """
        register a class for dumping loading
        - if it has attribute yaml_tag use that to register, else use class name
        - if it has methods to_yaml/from_yaml use those to dump/load else dump attributes
          as mapping
        """
        tag = getattr(cls, 'yaml_tag', '!' + cls.__name__)
        try:
            self.representer.add_representer(cls, cls.to_yaml)
        except AttributeError:

            def t_y(representer, data):
                return representer.represent_yaml_object(tag, data, cls, flow_style=representer.default_flow_style)
            self.representer.add_representer(cls, t_y)
        try:
            self.constructor.add_constructor(tag, cls.from_yaml)
        except AttributeError:

            def f_y(constructor, node):
                return constructor.construct_yaml_object(node, cls)
            self.constructor.add_constructor(tag, f_y)
        return cls

    def parse(self, stream):
        """
        Parse a YAML stream and produce parsing events.
        """
        _, parser = self.get_constructor_parser(stream)
        try:
            while parser.check_event():
                yield parser.get_event()
        finally:
            parser.dispose()
            try:
                self._reader.reset_reader()
            except AttributeError:
                pass
            try:
                self._scanner.reset_scanner()
            except AttributeError:
                pass

    def __enter__(self):
        self._context_manager = YAMLContextManager(self)
        return self

    def __exit__(self, typ, value, traceback):
        if typ:
            nprint('typ', typ)
        self._context_manager.teardown_output()
        self._context_manager = None

    def _indent(self, mapping=None, sequence=None, offset=None):
        if mapping is not None:
            self.map_indent = mapping
        if sequence is not None:
            self.sequence_indent = sequence
        if offset is not None:
            self.sequence_dash_offset = offset

    @property
    def indent(self):
        return self._indent

    @indent.setter
    def indent(self, val):
        self.old_indent = val

    @property
    def block_seq_indent(self):
        return self.sequence_dash_offset

    @block_seq_indent.setter
    def block_seq_indent(self, val):
        self.sequence_dash_offset = val

    def compact(self, seq_seq=None, seq_map=None):
        self.compact_seq_seq = seq_seq
        self.compact_seq_map = seq_map