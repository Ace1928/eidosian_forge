from . import screen
from . import FSM
import string
class term(screen.screen):
    """This class is an abstract, generic terminal.
    This does nothing. This is a placeholder that
    provides a common base class for other terminals
    such as an ANSI terminal. """

    def __init__(self, r=24, c=80, *args, **kwargs):
        screen.screen.__init__(self, r, c, *args, **kwargs)