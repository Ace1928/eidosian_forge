import pprint
import re
import typing as t
from markupsafe import Markup
from . import defaults
from . import nodes
from .environment import Environment
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .runtime import concat  # type: ignore
from .runtime import Context
from .runtime import Undefined
from .utils import import_string
from .utils import pass_context
def _install_callables(self, gettext: t.Callable[[str], str], ngettext: t.Callable[[str, str, int], str], newstyle: t.Optional[bool]=None, pgettext: t.Optional[t.Callable[[str, str], str]]=None, npgettext: t.Optional[t.Callable[[str, str, str, int], str]]=None) -> None:
    if newstyle is not None:
        self.environment.newstyle_gettext = newstyle
    if self.environment.newstyle_gettext:
        gettext = _make_new_gettext(gettext)
        ngettext = _make_new_ngettext(ngettext)
        if pgettext is not None:
            pgettext = _make_new_pgettext(pgettext)
        if npgettext is not None:
            npgettext = _make_new_npgettext(npgettext)
    self.environment.globals.update(gettext=gettext, ngettext=ngettext, pgettext=pgettext, npgettext=npgettext)