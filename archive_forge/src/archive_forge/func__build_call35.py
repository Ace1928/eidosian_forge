import builtins
import configparser
import operator
import sys
from cherrypy._cpcompat import text_or_bytes
def _build_call35(self, o):
    """
        Workaround for python 3.5 _ast.Call signature, docs found here
        https://greentreesnakes.readthedocs.org/en/latest/nodes.html
        """
    import ast
    callee = self.build(o.func)
    args = []
    if o.args is not None:
        for a in o.args:
            if isinstance(a, ast.Starred):
                args.append(self.build(a.value))
            else:
                args.append(self.build(a))
    kwargs = {}
    for kw in o.keywords:
        if kw.arg is None:
            rst = self.build(kw.value)
            if not isinstance(rst, dict):
                raise TypeError('Invalid argument for call.Must be a mapping object.')
            for k, v in rst.items():
                if k not in kwargs:
                    kwargs[k] = v
        else:
            kwargs[kw.arg] = self.build(kw.value)
    return callee(*args, **kwargs)