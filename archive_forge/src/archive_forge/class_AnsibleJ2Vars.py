from collections import ChainMap
from jinja2.utils import missing
from ansible.errors import AnsibleError, AnsibleUndefinedVariable
from ansible.module_utils.common.text.converters import to_native
class AnsibleJ2Vars(ChainMap):
    """Helper variable storage class that allows for nested variables templating: `foo: "{{ bar }}"`."""

    def __init__(self, templar, globals, locals=None):
        self._templar = templar
        super().__init__(_process_locals(locals), self._templar.available_variables, globals)

    def __getitem__(self, varname):
        variable = super().__getitem__(varname)
        from ansible.vars.hostvars import HostVars
        if varname == 'vars' and isinstance(variable, dict) or isinstance(variable, HostVars) or hasattr(variable, '__UNSAFE__'):
            return variable
        try:
            return self._templar.template(variable)
        except AnsibleUndefinedVariable as e:
            return self._templar.environment.undefined(hint=f'{variable}: {e.message}', name=varname, exc=AnsibleUndefinedVariable)
        except Exception as e:
            msg = getattr(e, 'message', None) or to_native(e)
            raise AnsibleError(f"An unhandled exception occurred while templating '{to_native(variable)}'. Error was a {type(e)}, original message: {msg}")

    def add_locals(self, locals):
        """If locals are provided, create a copy of self containing those
        locals in addition to what is already in this variable proxy.
        """
        if locals is None:
            return self
        current_locals = self.maps[0]
        current_globals = self.maps[2]
        new_locals = current_locals | locals
        return AnsibleJ2Vars(self._templar, current_globals, locals=new_locals)