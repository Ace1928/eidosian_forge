import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def QuoteForRspFile(arg, quote_cmd=True):
    """Quote a command line argument so that it appears as one argument when
    processed via cmd.exe and parsed by CommandLineToArgvW (as is typical for
    Windows programs)."""
    if arg.find('/') > 0 or arg.count('/') > 1:
        arg = os.path.normpath(arg)
    if quote_cmd:
        arg = windows_quoter_regex.sub(lambda mo: 2 * mo.group(1) + '\\"', arg)
    arg = arg.replace('%', '%%')
    if quote_cmd:
        return f'"{arg}"'
    return arg