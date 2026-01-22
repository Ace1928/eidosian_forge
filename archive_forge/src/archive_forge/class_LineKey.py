from typing import Optional
from ...public import PanelMetricsHelper, Run
from .runset import Runset
from .util import Attr, Base, Panel, nested_get, nested_set
class LineKey:

    def __init__(self, key: str) -> None:
        self.key = key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f'LineKey(key={self.key!r})'

    @classmethod
    def from_run(cls, run: 'Run', metric: str) -> 'LineKey':
        key = f'{run.id}:{metric}'
        return cls(key)

    @classmethod
    def from_panel_agg(cls, runset: 'Runset', panel: 'Panel', metric: str) -> 'LineKey':
        key = f'{runset.id}-config:group:{panel.groupby}:null:{metric}'
        return cls(key)

    @classmethod
    def from_runset_agg(cls, runset: 'Runset', metric: str) -> 'LineKey':
        groupby = runset.groupby
        if runset.groupby is None:
            groupby = 'null'
        key = f'{runset.id}-run:group:{groupby}:{metric}'
        return cls(key)