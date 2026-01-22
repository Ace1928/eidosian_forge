from __future__ import absolute_import, unicode_literals, print_function
import sys
import os
import warnings
import glob
from importlib import import_module
import ruamel.yaml
from ruamel.yaml.error import UnsafeLoaderWarning, YAMLError  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.loader import BaseLoader, SafeLoader, Loader, RoundTripLoader  # NOQA
from ruamel.yaml.dumper import BaseDumper, SafeDumper, Dumper, RoundTripDumper  # NOQA
from ruamel.yaml.compat import StringIO, BytesIO, with_metaclass, PY3, nprint
from ruamel.yaml.resolver import VersionedResolver, Resolver  # NOQA
from ruamel.yaml.representer import (
from ruamel.yaml.constructor import (
from ruamel.yaml.loader import Loader as UnsafeLoader
class XDumper(CEmitter, self.Representer, rslvr):

    def __init__(selfx, stream, default_style=None, default_flow_style=None, canonical=None, indent=None, width=None, allow_unicode=None, line_break=None, encoding=None, explicit_start=None, explicit_end=None, version=None, tags=None, block_seq_indent=None, top_level_colon_align=None, prefix_colon=None):
        CEmitter.__init__(selfx, stream, canonical=canonical, indent=indent, width=width, encoding=encoding, allow_unicode=allow_unicode, line_break=line_break, explicit_start=explicit_start, explicit_end=explicit_end, version=version, tags=tags)
        selfx._emitter = selfx._serializer = selfx._representer = selfx
        self.Representer.__init__(selfx, default_style=default_style, default_flow_style=default_flow_style)
        rslvr.__init__(selfx)