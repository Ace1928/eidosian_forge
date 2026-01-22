from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior
from kivy.uix.layout import Layout
from kivy.properties import (
from kivy.factory import Factory
def _catch_layout_trigger(self, instance=None, value=None):
    rv = self.recycleview
    if rv is None:
        return
    idx = self.view_indices.get(instance)
    if idx is not None:
        if self._size_needs_update:
            return
        opt = self.view_opts[idx]
        if instance.size == opt['size'] and instance.size_hint == opt['size_hint'] and (instance.size_hint_min == opt['size_hint_min']) and (instance.size_hint_max == opt['size_hint_max']) and (instance.pos_hint == opt['pos_hint']):
            return
        self._size_needs_update = True
        rv.refresh_from_layout(view_size=True)
    else:
        rv.refresh_from_layout()