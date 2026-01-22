from __future__ import annotations
from contextlib import suppress
from typing import TYPE_CHECKING
from warnings import warn
import numpy as np
from .._utils import to_rgba
from .._utils.registry import RegistryHierarchyMeta
from ..exceptions import PlotnineError, deprecated_themeable_name
from .elements import element_blank
from .elements.element_base import element_base
class plot_title(themeable):
    """
    Plot title

    Parameters
    ----------
    theme_element : element_text

    Notes
    -----
    The default horizontal alignment for the title is center. However the
    title will be left aligned if and only if there is a subtitle and its
    horizontal alignment has not been set (so it defaults to the left).

    The defaults ensure that, short titles are not awkwardly left-aligned,
    and that a title and a subtitle will not be awkwardly mis-aligned in
    the center or with different alignments.
    """
    _omit = ['margin']

    def apply_figure(self, figure: Figure, targets: ThemeTargets):
        super().apply_figure(figure, targets)
        if (text := targets.plot_title):
            props = self.properties
            with suppress(KeyError):
                del props['ha']
            text.set(**props)

    def blank_figure(self, figure: Figure, targets: ThemeTargets):
        super().blank_figure(figure, targets)
        if (text := targets.plot_title):
            text.set_visible(False)