from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
from triad import SerializableRLock
from triad.utils.convert import to_timedelta
from tune import Monitor, TrialReport, TrialReportLogger, parse_monitor
class NotebookSimpleChart(Monitor):

    def __init__(self, interval: Any='1sec', best_only: bool=True, always_update: bool=False):
        super().__init__()
        self._lock = SerializableRLock()
        self._last: Any = None
        self._bins: Dict[str, '_ReportBin'] = {}
        self._interval = to_timedelta(interval)
        self._best_only = best_only
        self._always_update = always_update

    def on_report(self, report: TrialReport) -> None:
        now = datetime.now()
        with self._lock:
            key = str(report.trial.keys)
            if key not in self._bins:
                self._bins[key] = _ReportBin(new_best_only=self._best_only)
            rbin = self._bins[key]
        updated = rbin.on_report(report)
        if not updated and (not self._always_update):
            return
        with self._lock:
            if self._last is None or now - self._last > self._interval:
                self._redraw()
                self._last = datetime.now()

    def plot(self, df: pd.DataFrame) -> None:
        return

    def finalize(self) -> None:
        self._redraw()

    def _redraw(self) -> None:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        df = pd.concat([pd.DataFrame(x.records, columns=['partition', 'rung', 'time', 'id', 'metric', 'best_metric']) for x in self._bins.values()])
        clear_output()
        self.plot(df)
        plt.show()
        for best in [x.best for x in self._bins.values() if x.best is not None]:
            if best is not None:
                print(best.trial.keys, best.metric, best)