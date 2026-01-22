import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
class HandlerRegistry(dict):
    GENERIC_OUTPUT_PREFERENCES = ['numpy', 'ctypesarrays']

    def __init__(self, plugin_match):
        self.match = plugin_match
        self.output_handler = None
        self.preferredOutput = None
        self.all_output_handlers = []

    def __call__(self, value):
        """Lookup of handler for given value"""
        try:
            typ = value.__class__
        except AttributeError:
            typ = type(value)
        handler = self.get(typ)
        if not handler:
            if hasattr(typ, '__mro__'):
                for base in typ.__mro__:
                    handler = self.get(base)
                    if not handler:
                        handler = self.match(base)
                        if handler:
                            handler = handler.load()
                            if handler:
                                handler = handler()
                    if handler:
                        self[typ] = handler
                        if hasattr(handler, 'registerEquivalent'):
                            handler.registerEquivalent(typ, base)
                        return handler
            print(self.keys())
            raise TypeError('No array-type handler for type %s.%s (value: %s) registered' % (typ.__module__, typ.__name__, repr(value)[:50]))
        return handler

    def handler_by_plugin_name(self, name):
        plugin = plugins.FormatHandler.by_name(name)
        if plugin:
            try:
                return plugin.load()
            except ImportError:
                return None
        else:
            raise RuntimeError('No handler of name %s found' % (name,))

    def get_output_handler(self):
        """Fast-path lookup for output handler object"""
        if self.output_handler is None:
            if self.preferredOutput is not None:
                self.output_handler = self.handler_by_plugin_name(self.preferredOutput)
            if not self.output_handler:
                for preferred in self.GENERIC_OUTPUT_PREFERENCES:
                    self.output_handler = self.handler_by_plugin_name(preferred)
                    if self.output_handler:
                        break
            if not self.output_handler:
                raise RuntimeError('Unable to find any output handler at all (not even ctypes/numpy ones!)')
        return self.output_handler

    def register(self, handler, types=None):
        """Register this class as handler for given set of types"""
        if not isinstance(types, (list, tuple)):
            types = [types]
        for type in types:
            self[type] = handler
        if handler.isOutput:
            self.all_output_handlers.append(handler)

    def registerReturn(self, handler):
        """Register this handler as the default return-type handler"""
        if isinstance(handler, (str, unicode)):
            self.preferredOutput = handler
            self.output_handler = None
        else:
            self.preferredOutput = None
            self.output_handler = handler