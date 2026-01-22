from __future__ import annotations
import typing
from copy import copy, deepcopy
from functools import cached_property
from typing import overload
from ..exceptions import PlotnineError
from ..options import get_option, set_option
from .targets import ThemeTargets
from .themeable import Themeables, themeable
def _smart_title_and_subtitle_ha(self):
    """
        Smartly add the horizontal alignment for the title and subtitle
        """
    from .elements import element_text
    has_title = bool(self.plot.labels.get('title', '')) and (not self.T.is_blank('plot_title'))
    has_subtitle = bool(self.plot.labels.get('subtitle', '')) and (not self.T.is_blank('plot_subtitle'))
    title_ha = self.getp(('plot_title', 'ha'))
    subtitle_ha = self.getp(('plot_subtitle', 'ha'))
    default_title_ha, default_subtitle_ha = ('center', 'left')
    kwargs = {}
    if has_title and title_ha is None:
        if has_subtitle and (not subtitle_ha):
            title_ha = default_subtitle_ha
        else:
            title_ha = default_title_ha
        kwargs['plot_title'] = element_text(ha=title_ha)
    if has_subtitle and subtitle_ha is None:
        subtitle_ha = default_subtitle_ha
        kwargs['plot_subtitle'] = element_text(ha=subtitle_ha)
    if kwargs:
        self += theme(**kwargs)