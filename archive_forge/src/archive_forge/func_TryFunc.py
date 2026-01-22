from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import functools
import itertools
import math
import random
import sys
import time
from googlecloudsdk.core import exceptions
def TryFunc():
    try:
        return (func(*args, **kwargs), None)
    except:
        return (None, sys.exc_info())