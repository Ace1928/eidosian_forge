import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
Find the list of patches.

    :param tree: Tree to read from
    