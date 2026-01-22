from __future__ import annotations
import logging # isort:skip
from ...core.enums import ToolIcon
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.bases import Init
from ...core.property.singletons import Intrinsic
from .ui_element import UIElement
class TablerIcon(Icon):
    """
    Icons from an external icon provider (https://tabler-icons.io/).

    .. note::
        This icon set is MIT licensed (see https://github.com/tabler/tabler-icons/blob/master/LICENSE).

    .. note::
        External icons are loaded from third-party servers and may not be available
        immediately (e.g. due to slow internet connection) or not available at all.
        It isn't possible to create a self-contained bundles with the use of
        ``inline`` resources. To circumvent this, one use ``SVGIcon``, by copying
        the SVG contents of an icon from Tabler's web site.

    """

    def __init__(self, icon_name: Init[str]=Intrinsic, **kwargs) -> None:
        super().__init__(icon_name=icon_name, **kwargs)
    icon_name = Required(String, help='\n    The name of the icon. See https://tabler-icons.io/ for the list of names.\n    ')