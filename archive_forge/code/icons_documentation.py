from __future__ import annotations
import logging # isort:skip
from ...core.enums import ToolIcon
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.bases import Init
from ...core.property.singletons import Intrinsic
from .ui_element import UIElement

    Icons from an external icon provider (https://tabler-icons.io/).

    .. note::
        This icon set is MIT licensed (see https://github.com/tabler/tabler-icons/blob/master/LICENSE).

    .. note::
        External icons are loaded from third-party servers and may not be available
        immediately (e.g. due to slow internet connection) or not available at all.
        It isn't possible to create a self-contained bundles with the use of
        ``inline`` resources. To circumvent this, one use ``SVGIcon``, by copying
        the SVG contents of an icon from Tabler's web site.

    