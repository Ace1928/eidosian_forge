from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
def NeedsSeparatingHyphenHyphen(self, flag='help'):
    """Returns whether a the trace need '--' before '--help'.

    '--' is needed when the component takes keyword arguments, when the value of
    flag matches one of the argument of the component, or the component takes in
    keyword-only arguments(e.g. argument with default value).

    Args:
      flag: the flag available for the trace

    Returns:
      True for needed '--', False otherwise.

    """
    element = self.GetLastHealthyElement()
    component = element.component
    spec = inspectutils.GetFullArgSpec(component)
    return spec.varkw is not None or flag in spec.args or flag in spec.kwonlyargs