from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import inspect
from fire import inspectutils
import six
def MemberVisible(component, name, member, class_attrs=None, verbose=False):
    """Returns whether a member should be included in auto-completion or help.

  Determines whether a member of an object with the specified name should be
  included in auto-completion or help text(both usage and detailed help).

  If the member name starts with '__', it will always be excluded. If it
  starts with only one '_', it will be included for all non-string types. If
  verbose is True, the members, including the private members, are included.

  When not in verbose mode, some modules and functions are excluded as well.

  Args:
    component: The component containing the member.
    name: The name of the member.
    member: The member itself.
    class_attrs: (optional) If component is a class, provide this as:
      inspectutils.GetClassAttrsDict(component). If not provided, it will be
      computed.
    verbose: Whether to include private members.
  Returns
    A boolean value indicating whether the member should be included.
  """
    if isinstance(name, six.string_types) and name.startswith('__'):
        return False
    if verbose:
        return True
    if member is absolute_import or member is division or member is print_function:
        return False
    if isinstance(member, type(absolute_import)) and six.PY34:
        return False
    if inspect.ismodule(member) and member is six:
        return False
    if inspect.isclass(component):
        if class_attrs is None:
            class_attrs = inspectutils.GetClassAttrsDict(class_attrs) or {}
        class_attr = class_attrs.get(name)
        if class_attr:
            if class_attr.kind in ('method', 'property'):
                return False
            tuplegetter = getattr(collections, '_tuplegetter', type(None))
            if isinstance(class_attr.object, tuplegetter):
                return False
    if six.PY2 and inspect.isfunction(component) and (name in ('func_closure', 'func_code', 'func_defaults', 'func_dict', 'func_doc', 'func_globals', 'func_name')):
        return False
    if six.PY2 and inspect.ismethod(component) and (name in ('im_class', 'im_func', 'im_self')):
        return False
    if isinstance(name, six.string_types):
        return not name.startswith('_')
    return True