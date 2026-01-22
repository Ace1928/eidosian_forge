import argparse
import base64
import contextlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from oslo_utils import strutils
import yaml
from ironicclient.common.i18n import _
from ironicclient import exc
def args_array_to_dict(kwargs, key_to_convert):
    """Convert the value in a dictionary entry to a dictionary.

    From the kwargs dictionary, converts the value of the key_to_convert
    entry from a list of key-value pairs to a dictionary.

    :param kwargs: a dictionary
    :param key_to_convert: the key (in kwargs), whose value is expected to
        be a list of key=value strings. This value will be converted to a
        dictionary.
    :returns: kwargs, the (modified) dictionary
    """
    values_to_convert = kwargs.get(key_to_convert)
    if values_to_convert:
        kwargs[key_to_convert] = key_value_pairs_to_dict(values_to_convert)
    return kwargs