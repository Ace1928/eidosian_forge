from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
def _set_view_adapter(self, value):
    view_adapter = self._view_adapter
    if value is view_adapter:
        return
    if view_adapter is not None:
        self._view_adapter = None
        view_adapter.detach_recycleview()
    if value is None:
        return True
    if not isinstance(value, RecycleDataAdapter):
        raise ValueError('Expected object based on RecycleAdapter, got {}'.format(value.__class__))
    self._view_adapter = value
    value.attach_recycleview(self)
    self.refresh_from_layout()
    return True