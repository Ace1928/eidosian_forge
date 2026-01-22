from __future__ import (absolute_import, division, print_function)
from jinja2.nativetypes import NativeTemplate
class AnsibleJ2Template(NativeTemplate):
    """
    A helper class, which prevents Jinja2 from running AnsibleJ2Vars through dict().
    Without this, {% include %} and similar will create new contexts unlike the special
    one created in Templar.template. This ensures they are all alike, except for
    potential locals.
    """

    def new_context(self, vars=None, shared=False, locals=None):
        if vars is None:
            vars = dict(self.globals or ())
        if isinstance(vars, dict):
            vars = vars.copy()
            if locals is not None:
                vars.update(locals)
        else:
            vars = vars.add_locals(locals)
        return self.environment.context_class(self.environment, vars, self.name, self.blocks)