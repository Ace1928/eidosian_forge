import re
import itertools
import textwrap
import uuid
import param
from param.display import register_display_accessor
from param._utils import async_executor
@magics_class
class ParamMagics(Magics):
    """
        Implements the %params line magic used to inspect the parameters
        of a parameterized class or object.
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_pager = ParamPager()

    @line_magic
    def params(self, parameter_s='', namespaces=None):
        """
            The %params line magic accepts a single argument which is a
            handle on the parameterized object to be inspected. If the
            object can be found in the active namespace, information about
            the object's parameters is displayed in the IPython pager.

            Usage: %params <parameterized class or object>
            """
        if parameter_s == '':
            print('Please specify an object to inspect.')
            return
        obj = self.shell._object_find(parameter_s)
        if obj.found is False:
            print('Object %r not found in the namespace.' % parameter_s)
            return
        page.page(self.param_pager(obj.obj))