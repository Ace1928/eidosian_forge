from __future__ import print_function, absolute_import
from textwrap import dedent
from shibokensupport.signature import inspect, typing
from shibokensupport.signature.mapping import ellipsis
from shibokensupport.signature.lib.tool import SimpleNamespace
def _valueerror(self, err_values):
    err_values = ', '.join(map(str, err_values))
    allowed_values = ', '.join(map(str, self.allowed_values))
    raise ValueError(dedent("            Not allowed: '{err_values}'.\n            The only allowed values are '{allowed_values}'.\n            ".format(**locals())))