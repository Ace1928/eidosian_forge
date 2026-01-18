import re
import sys
from IPython.utils import py3compat
from IPython.utils.encoding import get_stream_enc
from IPython.core.oinspect import OInfo
Do a full, attribute-walking lookup of the ifun in the various
        namespaces for the given IPython InteractiveShell instance.

        Return a dict with keys: {found, obj, ospace, ismagic}

        Note: can cause state changes because of calling getattr, but should
        only be run if autocall is on and if the line hasn't matched any
        other, less dangerous handlers.

        Does cache the results of the call, so can be called multiple times
        without worrying about *further* damaging state.
        