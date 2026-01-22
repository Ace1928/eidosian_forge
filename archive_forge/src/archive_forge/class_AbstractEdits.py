from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
class AbstractEdits:
    default_kwargs = {'line': 'hello world', 'cursor_offset': 5, 'cut_buffer': 'there'}

    def __init__(self, simple_edits=None, cut_buffer_edits=None):
        self.simple_edits = {} if simple_edits is None else simple_edits
        self.cut_buffer_edits = {} if cut_buffer_edits is None else cut_buffer_edits
        self.awaiting_config = {}

    def add(self, key, func, overwrite=False):
        if key in self:
            if overwrite:
                del self[key]
            else:
                raise ValueError(f'key {key!r} already has a mapping')
        params = getargspec(func)
        args = {k: v for k, v in self.default_kwargs.items() if k in params}
        r = func(**args)
        if len(r) == 2:
            if hasattr(func, 'kills'):
                raise ValueError('function %r returns two values, but has a kills attribute' % (func,))
            self.simple_edits[key] = func
        elif len(r) == 3:
            if not hasattr(func, 'kills'):
                raise ValueError('function %r returns three values, but has no kills attribute' % (func,))
            self.cut_buffer_edits[key] = func
        else:
            raise ValueError(f'return type of function {func!r} not recognized')

    def add_config_attr(self, config_attr, func):
        if config_attr in self.awaiting_config:
            raise ValueError(f'config attribute {config_attr!r} already has a mapping')
        self.awaiting_config[config_attr] = func

    def call(self, key, **kwargs):
        func = self[key]
        params = getargspec(func)
        args = {k: v for k, v in kwargs.items() if k in params}
        return func(**args)

    def call_without_cut(self, key, **kwargs):
        """Looks up the function and calls it, returning only line and cursor
        offset"""
        r = self.call_for_two(key, **kwargs)
        return r[:2]

    def __contains__(self, key):
        return key in self.simple_edits or key in self.cut_buffer_edits

    def __getitem__(self, key):
        if key in self.simple_edits:
            return self.simple_edits[key]
        if key in self.cut_buffer_edits:
            return self.cut_buffer_edits[key]
        raise KeyError(f'key {key!r} not mapped')

    def __delitem__(self, key):
        if key in self.simple_edits:
            del self.simple_edits[key]
        elif key in self.cut_buffer_edits:
            del self.cut_buffer_edits[key]
        else:
            raise KeyError(f'key {key!r} not mapped')