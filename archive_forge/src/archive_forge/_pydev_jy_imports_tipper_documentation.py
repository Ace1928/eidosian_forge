import traceback
from io import StringIO
from java.lang import StringBuffer  # @UnresolvedImport
from java.lang import String  # @UnresolvedImport
import java.lang  # @UnresolvedImport
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from org.python.core import PyReflectedFunction  # @UnresolvedImport
from org.python import core  # @UnresolvedImport
from org.python.core import PyClass  # @UnresolvedImport
import java.util
Get information about the arguments accepted by a code object.

                Three things are returned: (args, varargs, varkw), where 'args' is
                a list of argument names (possibly containing nested lists), and
                'varargs' and 'varkw' are the names of the * and ** arguments or None.