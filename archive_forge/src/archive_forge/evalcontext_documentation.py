from io import StringIO
import traceback
import threading
import pdb
import sys

    A specialized version of the python debugger that redirects stdout
    to a given stream when interacting with the user.  Stdout is *not*
    redirected when traced code is executed.
    