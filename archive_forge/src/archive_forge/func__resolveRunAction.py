import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def _resolveRunAction(self, interactiveFunction, functionGroup, functionTip):
    if isinstance(functionGroup, ActionGroupParameter):
        functionGroup.setButtonOpts(visible=True)
        child = None
    else:
        createOpts = self._makePopulatedActionTemplate(interactiveFunction.__name__, functionTip)
        child = Parameter.create(**createOpts)
        child.sigActivated.connect(interactiveFunction.runFromAction)
        if functionGroup:
            functionGroup.addChild(child, existOk=self.existOk)
    return child