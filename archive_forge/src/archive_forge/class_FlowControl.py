import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.lib import fsm
from .settings_static import SettingsStatic
class FlowControl:
    _fsm: fsm.FsmWithContext['Record', StateContext]

    def __init__(self, settings: SettingsStatic, forward_record: Callable[['Record'], None], write_record: Callable[['Record'], int], pause_marker: Callable[[], None], recover_records: Callable[[int, int], None], _threshold_bytes_high: int=0, _threshold_bytes_mid: int=0, _threshold_bytes_low: int=0) -> None:
        if _threshold_bytes_high == 0 or _threshold_bytes_mid == 0 or _threshold_bytes_low == 0:
            threshold = settings._network_buffer or DEFAULT_THRESHOLD
            _threshold_bytes_high = threshold
            _threshold_bytes_mid = threshold // 2
            _threshold_bytes_low = threshold // 4
        assert _threshold_bytes_high > _threshold_bytes_mid > _threshold_bytes_low
        state_forwarding = StateForwarding(forward_record=forward_record, pause_marker=pause_marker, threshold_pause=_threshold_bytes_high)
        state_pausing = StatePausing(forward_record=forward_record, recover_records=recover_records, threshold_recover=_threshold_bytes_mid, threshold_forward=_threshold_bytes_low)
        self._fsm = fsm.FsmWithContext(states=[state_forwarding, state_pausing], table={StateForwarding: [fsm.FsmEntry(state_forwarding._should_pause, StatePausing, state_forwarding._pause)], StatePausing: [fsm.FsmEntry(state_pausing._should_unpause, StateForwarding, state_pausing._unpause), fsm.FsmEntry(state_pausing._should_recover, StatePausing, state_pausing._recover), fsm.FsmEntry(state_pausing._should_quiesce, StatePausing, state_pausing._quiesce)]})

    def flush(self) -> None:
        pass

    def flow(self, record: 'Record') -> None:
        self._fsm.input(record)