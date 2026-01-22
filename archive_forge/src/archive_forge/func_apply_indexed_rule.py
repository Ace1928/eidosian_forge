import logging
import sys
from pyomo.common.deprecation import relocated_module_attribute
def apply_indexed_rule(obj, rule, model, index, options=None):
    try:
        if options is None:
            if index.__class__ is tuple:
                return rule(model, *index)
            elif index is None and (not obj.is_indexed()):
                return rule(model)
            else:
                return rule(model, index)
        elif index.__class__ is tuple:
            return rule(model, *index, **options)
        elif index is None and (not obj.is_indexed()):
            return rule(model, **options)
        else:
            return rule(model, index, **options)
    except TypeError:
        try:
            if options is None:
                return rule(model)
            else:
                return rule(model, **options)
        except:
            if options is None:
                if index.__class__ is tuple:
                    return rule(model, *index)
                elif index is None and (not obj.is_indexed()):
                    return rule(model)
                else:
                    return rule(model, index)
            elif index.__class__ is tuple:
                return rule(model, *index, **options)
            elif index is None and (not obj.is_indexed()):
                return rule(model, **options)
            else:
                return rule(model, index, **options)