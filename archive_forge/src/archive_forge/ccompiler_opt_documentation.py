import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap

            mingw can be treated as a gcc, and also xlc even if it based on clang,
            but still has the same gcc optimization flags.
            