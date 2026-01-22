from logging import error
import io
import os
from pprint import pformat
import sys
from warnings import warn
from traitlets.utils.importstring import import_item
from IPython.core import magic_arguments, page
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, magic_escapes
from IPython.utils.text import format_screen, dedent, indent
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.ipstruct import Struct

        Allow to change the status of the autoawait option.

        This allow you to set a specific asynchronous code runner.

        If no value is passed, print the currently used asynchronous integration
        and whether it is activated.

        It can take a number of value evaluated in the following order:

        - False/false/off deactivate autoawait integration
        - True/true/on activate autoawait integration using configured default
          loop
        - asyncio/curio/trio activate autoawait integration and use integration
          with said library.

        - `sync` turn on the pseudo-sync integration (mostly used for
          `IPython.embed()` which does not run IPython with a real eventloop and
          deactivate running asynchronous code. Turning on Asynchronous code with
          the pseudo sync loop is undefined behavior and may lead IPython to crash.

        If the passed parameter does not match any of the above and is a python
        identifier, get said object from user namespace and set it as the
        runner, and activate autoawait.

        If the object is a fully qualified object name, attempt to import it and
        set it as the runner, and activate autoawait.

        The exact behavior of autoawait is experimental and subject to change
        across version of IPython and Python.
        