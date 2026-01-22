import functools
import os
import sys
import re
import shutil
import types
from .encoding import DEFAULT_ENCODING
import platform
def MethodType(func, instance):
    return types.MethodType(func, instance, type(instance))