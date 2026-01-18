from __future__ import print_function
import sys
import os
import platform
import io
import getopt
import re
import string
import errno
import copy
import glob
from jsbeautifier.__version__ import __version__
from jsbeautifier.javascript.options import BeautifierOptions
from jsbeautifier.javascript.beautifier import Beautifier
def write_beautified_output(pretty, local_options, outfile):
    if outfile == 'stdout':
        stream = sys.stdout
        if platform.platform().lower().startswith('windows'):
            if sys.version_info.major >= 3:
                stream = io.TextIOWrapper(sys.stdout.buffer, newline='')
            elif platform.architecture()[0] == '32bit':
                import msvcrt
                msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
            else:
                raise Exception('Pipe to stdout not supported on Windows with Python 2.x 64-bit.')
        stream.write(pretty)
    elif isFileDifferent(outfile, pretty):
        mkdir_p(os.path.dirname(outfile))
        with io.open(outfile, 'wt', newline='', encoding='UTF-8') as f:
            if not local_options.keep_quiet:
                print('beautified ' + outfile, file=sys.stdout)
            try:
                f.write(pretty)
            except TypeError:
                six = __import__('six')
                f.write(six.u(pretty))
    elif not local_options.keep_quiet:
        print('beautified ' + outfile + ' - unchanged', file=sys.stdout)