import argparse as argparse_mod
import collections
import copy
import errno
import json
import os
import re
import sys
import typing as ty
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import loading
import platformdirs
import yaml
from openstack import _log
from openstack.config import _util
from openstack.config import cloud_region
from openstack.config import defaults
from openstack.config import vendors
from openstack import exceptions
from openstack import warnings as os_warnings
def _fix_argv(argv):
    processed = collections.defaultdict(set)
    for index in range(0, len(argv)):
        if re.match('^--.*(_|-)+.*', argv[index]):
            split_args = argv[index].split('=')
            orig = split_args[0]
            new = orig.replace('_', '-')
            if orig != new:
                split_args[0] = new
                argv[index] = '='.join(split_args)
            processed[new].add(orig)
    overlap: ty.List[str] = []
    for new, old in processed.items():
        if len(old) > 1:
            overlap.extend(old)
    if overlap:
        raise exceptions.ConfigException("The following options were given: '{options}' which contain duplicates except that one has _ and one has -. There is no sane way for us to know what you're doing. Remove the duplicate option and try again".format(options=','.join(overlap)))