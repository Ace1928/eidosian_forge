import errno
import os
import re
import subprocess
import sys
import glob
def JoinPath(*args):
    return os.path.normpath(os.path.join(*args))