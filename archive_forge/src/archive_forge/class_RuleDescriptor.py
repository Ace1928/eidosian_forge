from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
class RuleDescriptor(SimpleDescriptor):
    """Represents the rule descriptor element: a set of glyph substitutions to
    trigger conditionally in some parts of the designspace.

    .. code:: python

        r1 = RuleDescriptor()
        r1.name = "unique.rule.name"
        r1.conditionSets.append([dict(name="weight", minimum=-10, maximum=10), dict(...)])
        r1.conditionSets.append([dict(...), dict(...)])
        r1.subs.append(("a", "a.alt"))

    .. code:: xml

        <!-- optional: list of substitution rules -->
        <rules>
            <rule name="vertical.bars">
                <conditionset>
                    <condition minimum="250.000000" maximum="750.000000" name="weight"/>
                    <condition minimum="100" name="width"/>
                    <condition minimum="10" maximum="40" name="optical"/>
                </conditionset>
                <sub name="cent" with="cent.alt"/>
                <sub name="dollar" with="dollar.alt"/>
            </rule>
        </rules>
    """
    _attrs = ['name', 'conditionSets', 'subs']

    def __init__(self, *, name=None, conditionSets=None, subs=None):
        self.name = name
        'string. Unique name for this rule. Can be used to reference this rule data.'
        self.conditionSets = conditionSets or []
        'a list of conditionsets.\n\n        -  Each conditionset is a list of conditions.\n        -  Each condition is a dict with ``name``, ``minimum`` and ``maximum`` keys.\n        '
        self.subs = subs or []
        'list of substitutions.\n\n        -  Each substitution is stored as tuples of glyphnames, e.g. ("a", "a.alt").\n        -  Note: By default, rules are applied first, before other text\n           shaping/OpenType layout, as they are part of the\n           `Required Variation Alternates OpenType feature <https://docs.microsoft.com/en-us/typography/opentype/spec/features_pt#-tag-rvrn>`_.\n           See ref:`rules-element` § Attributes.\n        '