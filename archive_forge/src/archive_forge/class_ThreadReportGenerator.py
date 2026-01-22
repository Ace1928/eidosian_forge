import gc
import sys
import threading
from oslo_reports.models import threading as tm
from oslo_reports.models import with_default_views as mwdv
from oslo_reports.views.text import generic as text_views
class ThreadReportGenerator(object):
    """A Thread Data Generator

    This generator returns a collection of
    :class:`oslo_reports.models.threading.ThreadModel`
    objects by introspecting the current python state using
    :func:`sys._current_frames()` .  Its constructor may optionally
    be passed a frame object.  This frame object will be interpreted
    as the actual stack trace for the current thread, and, come generation
    time, will be used to replace the stack trace of the thread in which
    this code is running.
    """

    def __init__(self, curr_thread_traceback=None):
        self.traceback = curr_thread_traceback

    def __call__(self):
        threadModels = dict(((thread_id, tm.ThreadModel(thread_id, stack)) for thread_id, stack in sys._current_frames().items()))
        if self.traceback is not None:
            curr_thread_id = threading.current_thread().ident
            threadModels[curr_thread_id] = tm.ThreadModel(curr_thread_id, self.traceback)
        return mwdv.ModelWithDefaultViews(threadModels, text_view=text_views.MultiView())