import inspect
import textwrap
class TemplateExpressionError(ValueError):
    """Special ValueError raised by getitem for template arguments

    This exception is triggered by the Pyomo expression system when
    attempting to get a member of an IndexedComponent using either a
    TemplateIndex, or an expression containing a TemplateIndex.

    Users should never see this exception.

    """

    def __init__(self, template, *args, **kwds):
        self.template = template
        super(TemplateExpressionError, self).__init__(*args, **kwds)