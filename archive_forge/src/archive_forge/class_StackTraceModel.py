import traceback
from oslo_reports.models import with_default_views as mwdv
from oslo_reports.views.text import threading as text_views
class StackTraceModel(mwdv.ModelWithDefaultViews):
    """A Stack Trace Model

    This model holds data from a python stack trace,
    commonly extracted from running thread information

    :param stack_state: the python stack_state object
    """

    def __init__(self, stack_state):
        super(StackTraceModel, self).__init__(text_view=text_views.StackTraceView())
        if stack_state is not None:
            self['lines'] = [{'filename': fn, 'line': ln, 'name': nm, 'code': cd} for fn, ln, nm, cd in traceback.extract_stack(stack_state)]
            if getattr(stack_state, 'f_exc_type', None) is not None:
                self['root_exception'] = {'type': stack_state.f_exc_type, 'value': stack_state.f_exc_value}
            else:
                self['root_exception'] = None
        else:
            self['lines'] = []
            self['root_exception'] = None