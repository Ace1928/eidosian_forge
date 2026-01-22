import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def _format_log(self, log, in_filename, out_filename):
    text = '-' * 80 + '\n'
    text += 'Processing file %r\n outputting to %r\n' % (in_filename, out_filename)
    text += '-' * 80 + '\n\n'
    text += '\n'.join(log) + '\n'
    text += '-' * 80 + '\n\n'
    return text