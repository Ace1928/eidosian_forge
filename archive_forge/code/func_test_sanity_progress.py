import contextlib
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import task
from taskflow import test
from taskflow.utils import persistence_utils as p_utils
def test_sanity_progress(self):
    fired_events = []

    def notify_me(event_type, details):
        fired_events.append(details.pop('progress'))
    ev_count = 5
    t = ProgressTask('test', ev_count)
    t.notifier.register(task.EVENT_UPDATE_PROGRESS, notify_me)
    flo = lf.Flow('test')
    flo.add(t)
    e = self._make_engine(flo)
    e.run()
    self.assertEqual(ev_count + 1, len(fired_events))
    self.assertEqual(1.0, fired_events[-1])
    self.assertEqual(0.0, fired_events[0])