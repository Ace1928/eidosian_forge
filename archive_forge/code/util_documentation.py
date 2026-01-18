from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import time
import numpy as np
import six
import tensorflow as tf
Creates a SessionRunHook that initializes all passed iterators.