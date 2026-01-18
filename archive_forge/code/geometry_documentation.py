from __future__ import annotations
import logging # isort:skip
from math import inf
from ...core.enums import (
from ...core.properties import (
from ...core.property_aliases import BorderRadius
from ...core.property_mixins import (
from ..common.properties import Coordinate
from ..nodes import BoxNodes, Node
from .annotation import Annotation, DataAnnotation
from .arrows import ArrowHead, TeeHead
 Render a whisker along a dimension.

    See :ref:`ug_basic_annotations_whiskers` for information on plotting whiskers.

    