import traceback
from oslo_reports.models import with_default_views as mwdv
from oslo_reports.views.text import threading as text_views
class GreenThreadModel(mwdv.ModelWithDefaultViews):
    """A Green Thread Model

    This model holds data for information about an
    individual thread.  Unlike the thread model,
    it holds just a stack trace, since green threads
    do not have thread ids.

    .. seealso::

        Class :class:`StackTraceModel`

    :param stack: the python stack state for the green thread
    """

    def __init__(self, stack):
        super(GreenThreadModel, self).__init__({'stack_trace': StackTraceModel(stack)}, text_view=text_views.GreenThreadView())