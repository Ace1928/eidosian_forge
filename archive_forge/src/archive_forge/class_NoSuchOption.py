import typing as t
from gettext import gettext as _
from gettext import ngettext
from ._compat import get_text_stderr
from .utils import echo
from .utils import format_filename
class NoSuchOption(UsageError):
    """Raised if click attempted to handle an option that does not
    exist.

    .. versionadded:: 4.0
    """

    def __init__(self, option_name: str, message: t.Optional[str]=None, possibilities: t.Optional[t.Sequence[str]]=None, ctx: t.Optional['Context']=None) -> None:
        if message is None:
            message = _('No such option: {name}').format(name=option_name)
        super().__init__(message, ctx)
        self.option_name = option_name
        self.possibilities = possibilities

    def format_message(self) -> str:
        if not self.possibilities:
            return self.message
        possibility_str = ', '.join(sorted(self.possibilities))
        suggest = ngettext('Did you mean {possibility}?', '(Possible options: {possibilities})', len(self.possibilities)).format(possibility=possibility_str, possibilities=possibility_str)
        return f'{self.message} {suggest}'