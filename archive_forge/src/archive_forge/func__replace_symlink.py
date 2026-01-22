from __future__ import with_statement
import logging
import optparse
import os
import os.path
import re
import shutil
import subprocess
import sys
import itertools
def _replace_symlink(filename, newtarget):
    tmpfn = '%s.new' % filename
    os.symlink(newtarget, tmpfn)
    os.rename(tmpfn, filename)