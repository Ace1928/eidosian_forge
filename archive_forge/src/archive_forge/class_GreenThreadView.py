from oslo_reports.views import jinja_view as jv
class GreenThreadView(object):
    """A Green Thread View

    This view displays a green thread provided by the data
    model :class:`oslo_reports.models.threading.GreenThreadModel`
    """
    FORMAT_STR = '------{thread_str: ^60}------' + '\n' + '{stack_trace}'

    def __call__(self, model):
        return self.FORMAT_STR.format(thread_str=' Green Thread ', stack_trace=model.stack_trace)