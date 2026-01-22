import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecStamp(self, path):
    """Simple stamp command."""
    open(path, 'w').close()