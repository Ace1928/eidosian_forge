from __future__ import annotations
import difflib
import sys
import weakref
from typing import (
import param
from bokeh.models import CustomJS, LayoutDOM, Model as BkModel
from .io.datamodel import create_linked_datamodel
from .io.loading import LOADING_INDICATOR_CSS_CLASS
from .models import ReactiveHTML
from .reactive import Reactive
from .util.warnings import warn
from .viewable import Viewable
def _get_specs(self, link: 'Link', source: 'Reactive', target: 'JSLinkTarget') -> Sequence[Tuple['SourceModelSpec', 'TargetModelSpec', str | None]]:
    if link.code:
        return super()._get_specs(link, source, target)
    specs = []
    for src_spec, tgt_spec in link.properties.items():
        src_specs = src_spec.split('.')
        if len(src_specs) > 1:
            src_spec = ('.'.join(src_specs[:-1]), src_specs[-1])
        else:
            src_prop = src_specs[0]
            if isinstance(source, Reactive):
                src_prop = source._rename.get(src_prop, src_prop)
            src_spec = (None, src_prop)
        tgt_specs = tgt_spec.split('.')
        if len(tgt_specs) > 1:
            tgt_spec = ('.'.join(tgt_specs[:-1]), tgt_specs[-1])
        else:
            tgt_prop = tgt_specs[0]
            if isinstance(target, Reactive):
                tgt_prop = target._rename.get(tgt_prop, tgt_prop)
            tgt_spec = (None, tgt_prop)
        specs.append((src_spec, tgt_spec, None))
    return specs