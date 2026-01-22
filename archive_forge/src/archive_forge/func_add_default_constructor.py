from __future__ import annotations
import datetime
from datetime import timedelta as TimeDelta
import binascii
import sys
import types
import warnings
from collections.abc import Hashable, MutableSequence, MutableMapping
from ruamel.yaml.error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from ruamel.yaml.nodes import *                               # NOQA
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import (builtins_module, # NOQA
from ruamel.yaml.compat import ordereddict
from ruamel.yaml.tag import Tag
from ruamel.yaml.comments import *                               # NOQA
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from ruamel.yaml.scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
from ruamel.yaml.util import timestamp_regexp, create_timestamp
@classmethod
def add_default_constructor(cls, tag: str, method: Any=None, tag_base: str='tag:yaml.org,2002:python/') -> None:
    if not tag.startswith('tag:'):
        if method is None:
            method = 'construct_yaml_' + tag
        tag = tag_base + tag
    cls.add_constructor(tag, getattr(cls, method))