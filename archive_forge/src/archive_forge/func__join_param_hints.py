import typing as t
from gettext import gettext as _
from gettext import ngettext
from ._compat import get_text_stderr
from .utils import echo
from .utils import format_filename
def _join_param_hints(param_hint: t.Optional[t.Union[t.Sequence[str], str]]) -> t.Optional[str]:
    if param_hint is not None and (not isinstance(param_hint, str)):
        return ' / '.join((repr(x) for x in param_hint))
    return param_hint