from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
class PrefilterManager(Configurable):
    """Main prefilter component.

    The IPython prefilter is run on all user input before it is run.  The
    prefilter consumes lines of input and produces transformed lines of
    input.

    The implementation consists of two phases:

    1. Transformers
    2. Checkers and handlers

    Over time, we plan on deprecating the checkers and handlers and doing
    everything in the transformers.

    The transformers are instances of :class:`PrefilterTransformer` and have
    a single method :meth:`transform` that takes a line and returns a
    transformed line.  The transformation can be accomplished using any
    tool, but our current ones use regular expressions for speed.

    After all the transformers have been run, the line is fed to the checkers,
    which are instances of :class:`PrefilterChecker`.  The line is passed to
    the :meth:`check` method, which either returns `None` or a
    :class:`PrefilterHandler` instance.  If `None` is returned, the other
    checkers are tried.  If an :class:`PrefilterHandler` instance is returned,
    the line is passed to the :meth:`handle` method of the returned
    handler and no further checkers are tried.

    Both transformers and checkers have a `priority` attribute, that determines
    the order in which they are called.  Smaller priorities are tried first.

    Both transformers and checkers also have `enabled` attribute, which is
    a boolean that determines if the instance is used.

    Users or developers can change the priority or enabled attribute of
    transformers or checkers, but they must call the :meth:`sort_checkers`
    or :meth:`sort_transformers` method after changing the priority.
    """
    multi_line_specials = Bool(True).tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)

    def __init__(self, shell=None, **kwargs):
        super(PrefilterManager, self).__init__(shell=shell, **kwargs)
        self.shell = shell
        self._transformers = []
        self.init_handlers()
        self.init_checkers()

    def sort_transformers(self):
        """Sort the transformers by priority.

        This must be called after the priority of a transformer is changed.
        The :meth:`register_transformer` method calls this automatically.
        """
        self._transformers.sort(key=lambda x: x.priority)

    @property
    def transformers(self):
        """Return a list of checkers, sorted by priority."""
        return self._transformers

    def register_transformer(self, transformer):
        """Register a transformer instance."""
        if transformer not in self._transformers:
            self._transformers.append(transformer)
            self.sort_transformers()

    def unregister_transformer(self, transformer):
        """Unregister a transformer instance."""
        if transformer in self._transformers:
            self._transformers.remove(transformer)

    def init_checkers(self):
        """Create the default checkers."""
        self._checkers = []
        for checker in _default_checkers:
            checker(shell=self.shell, prefilter_manager=self, parent=self)

    def sort_checkers(self):
        """Sort the checkers by priority.

        This must be called after the priority of a checker is changed.
        The :meth:`register_checker` method calls this automatically.
        """
        self._checkers.sort(key=lambda x: x.priority)

    @property
    def checkers(self):
        """Return a list of checkers, sorted by priority."""
        return self._checkers

    def register_checker(self, checker):
        """Register a checker instance."""
        if checker not in self._checkers:
            self._checkers.append(checker)
            self.sort_checkers()

    def unregister_checker(self, checker):
        """Unregister a checker instance."""
        if checker in self._checkers:
            self._checkers.remove(checker)

    def init_handlers(self):
        """Create the default handlers."""
        self._handlers = {}
        self._esc_handlers = {}
        for handler in _default_handlers:
            handler(shell=self.shell, prefilter_manager=self, parent=self)

    @property
    def handlers(self):
        """Return a dict of all the handlers."""
        return self._handlers

    def register_handler(self, name, handler, esc_strings):
        """Register a handler instance by name with esc_strings."""
        self._handlers[name] = handler
        for esc_str in esc_strings:
            self._esc_handlers[esc_str] = handler

    def unregister_handler(self, name, handler, esc_strings):
        """Unregister a handler instance by name with esc_strings."""
        try:
            del self._handlers[name]
        except KeyError:
            pass
        for esc_str in esc_strings:
            h = self._esc_handlers.get(esc_str)
            if h is handler:
                del self._esc_handlers[esc_str]

    def get_handler_by_name(self, name):
        """Get a handler by its name."""
        return self._handlers.get(name)

    def get_handler_by_esc(self, esc_str):
        """Get a handler by its escape string."""
        return self._esc_handlers.get(esc_str)

    def prefilter_line_info(self, line_info):
        """Prefilter a line that has been converted to a LineInfo object.

        This implements the checker/handler part of the prefilter pipe.
        """
        handler = self.find_handler(line_info)
        return handler.handle(line_info)

    def find_handler(self, line_info):
        """Find a handler for the line_info by trying checkers."""
        for checker in self.checkers:
            if checker.enabled:
                handler = checker.check(line_info)
                if handler:
                    return handler
        return self.get_handler_by_name('normal')

    def transform_line(self, line, continue_prompt):
        """Calls the enabled transformers in order of increasing priority."""
        for transformer in self.transformers:
            if transformer.enabled:
                line = transformer.transform(line, continue_prompt)
        return line

    def prefilter_line(self, line, continue_prompt=False):
        """Prefilter a single input line as text.

        This method prefilters a single line of text by calling the
        transformers and then the checkers/handlers.
        """
        self.shell._last_input_line = line
        if not line:
            return ''
        if not continue_prompt or (continue_prompt and self.multi_line_specials):
            line = self.transform_line(line, continue_prompt)
        line_info = LineInfo(line, continue_prompt)
        stripped = line.strip()
        normal_handler = self.get_handler_by_name('normal')
        if not stripped:
            return normal_handler.handle(line_info)
        if continue_prompt and (not self.multi_line_specials):
            return normal_handler.handle(line_info)
        prefiltered = self.prefilter_line_info(line_info)
        return prefiltered

    def prefilter_lines(self, lines, continue_prompt=False):
        """Prefilter multiple input lines of text.

        This is the main entry point for prefiltering multiple lines of
        input.  This simply calls :meth:`prefilter_line` for each line of
        input.

        This covers cases where there are multiple lines in the user entry,
        which is the case when the user goes back to a multiline history
        entry and presses enter.
        """
        llines = lines.rstrip('\n').split('\n')
        if len(llines) > 1:
            out = '\n'.join([self.prefilter_line(line, lnum > 0) for lnum, line in enumerate(llines)])
        else:
            out = self.prefilter_line(llines[0], continue_prompt)
        return out