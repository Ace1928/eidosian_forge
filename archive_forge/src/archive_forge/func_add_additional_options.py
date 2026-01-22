from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
def add_additional_options(self, lang, parent_node, file_args):
    args = []
    for arg in file_args[lang].to_native():
        if self.is_argument_with_msbuild_xml_entry(arg):
            continue
        if arg == '%(AdditionalOptions)':
            args.append(arg)
        else:
            args.append(self.escape_additional_option(arg))
    ET.SubElement(parent_node, 'AdditionalOptions').text = ' '.join(args)