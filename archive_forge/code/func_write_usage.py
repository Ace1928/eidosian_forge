import typing as t
from contextlib import contextmanager
from gettext import gettext as _
from ._compat import term_len
from .parser import split_opt
def write_usage(self, prog: str, args: str='', prefix: t.Optional[str]=None) -> None:
    """Writes a usage line into the buffer.

        :param prog: the program name.
        :param args: whitespace separated list of arguments.
        :param prefix: The prefix for the first line. Defaults to
            ``"Usage: "``.
        """
    if prefix is None:
        prefix = f'{_('Usage:')} '
    usage_prefix = f'{prefix:>{self.current_indent}}{prog} '
    text_width = self.width - self.current_indent
    if text_width >= term_len(usage_prefix) + 20:
        indent = ' ' * term_len(usage_prefix)
        self.write(wrap_text(args, text_width, initial_indent=usage_prefix, subsequent_indent=indent))
    else:
        self.write(usage_prefix)
        self.write('\n')
        indent = ' ' * (max(self.current_indent, term_len(prefix)) + 4)
        self.write(wrap_text(args, text_width, initial_indent=indent, subsequent_indent=indent))
    self.write('\n')