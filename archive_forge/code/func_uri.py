import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
@property
def uri(self):
    """The URI for this :class:`.Namespace`'s template.

        I.e. whatever was sent to :meth:`.TemplateLookup.get_template()`.

        This is the equivalent of :attr:`.Template.uri`.

        """
    return self.template.uri