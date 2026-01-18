from __future__ import annotations
from typing import TYPE_CHECKING
import trio
def test_the_trio_scheduler_is_deterministic_if_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trio._core._run, '_ALLOW_DETERMINISTIC_SCHEDULING', True)
    traces = []
    for _ in range(10):
        state = trio._core._run._r.getstate()
        try:
            trio._core._run._r.seed(0)
            traces.append(trio.run(scheduler_trace))
        finally:
            trio._core._run._r.setstate(state)
    assert len(traces) == 10
    assert len(set(traces)) == 1