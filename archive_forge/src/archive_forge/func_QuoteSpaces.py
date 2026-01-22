import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def QuoteSpaces(s, quote='\\ '):
    return s.replace(' ', quote)