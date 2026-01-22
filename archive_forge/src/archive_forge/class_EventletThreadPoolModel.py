import futurist
from oslo_log import log as logging
from glance.i18n import _LE
class EventletThreadPoolModel(ThreadPoolModel):
    """A ThreadPoolModel suitable for use with evenlet/greenthreads."""
    DEFAULTSIZE = 1024

    @staticmethod
    def get_threadpool_executor_class():
        return futurist.GreenThreadPoolExecutor