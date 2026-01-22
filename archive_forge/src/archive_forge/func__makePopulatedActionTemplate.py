import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def _makePopulatedActionTemplate(self, functionName='', functionTip=None):
    createOpts = self.runActionTemplate.copy()
    defaultName = createOpts.get('defaultName', 'Run')
    name = defaultName if self.nest else functionName
    createOpts.setdefault('name', name)
    if functionTip:
        createOpts.setdefault('tip', functionTip)
    return createOpts