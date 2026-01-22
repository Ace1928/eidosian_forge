from oslo_reports.views import jinja_view as jv
class ThreadView(object):
    """A Thread Collection View

    This view displays a python thread provided by the data
    model :class:`oslo_reports.models.threading.ThreadModel`  # noqa
    """
    FORMAT_STR = '------{thread_str: ^60}------' + '\n' + '{stack_trace}'

    def __call__(self, model):
        return self.FORMAT_STR.format(thread_str=' Thread #{0} '.format(model.thread_id), stack_trace=model.stack_trace)