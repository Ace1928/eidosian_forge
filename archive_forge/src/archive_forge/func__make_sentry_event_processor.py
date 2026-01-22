from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
def _make_sentry_event_processor(self):

    def event_processor(event, _):
        if self.method_route is not None:
            event['transaction'] = sentry_transaction_from_function(self.method_route.func)
        return event
    return event_processor