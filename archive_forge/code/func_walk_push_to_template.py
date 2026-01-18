import textwrap
import pkgutil
import copy
import os
import json
from functools import reduce
def walk_push_to_template(fig_obj, template_obj, skip):
    """
    Move style properties from fig_obj to template_obj.

    Parameters
    ----------
    fig_obj: plotly.basedatatypes.BasePlotlyType
    template_obj: plotly.basedatatypes.BasePlotlyType
    skip: set of str
        Set of names of properties to skip
    """
    from _plotly_utils.basevalidators import CompoundValidator, CompoundArrayValidator, is_array
    for prop in list(fig_obj._props):
        if prop == 'template' or prop in skip:
            continue
        fig_val = fig_obj[prop]
        template_val = template_obj[prop]
        validator = fig_obj._get_validator(prop)
        if isinstance(validator, CompoundValidator):
            walk_push_to_template(fig_val, template_val, skip)
            if not fig_val._props:
                fig_obj[prop] = None
        elif isinstance(validator, CompoundArrayValidator) and fig_val:
            template_elements = list(template_val)
            template_element_names = [el.name for el in template_elements]
            template_propdefaults = template_obj[prop[:-1] + 'defaults']
            for fig_el in fig_val:
                element_name = fig_el.name
                if element_name:
                    skip = set()
                    if fig_el.name in template_element_names:
                        item_index = template_element_names.index(fig_el.name)
                        template_el = template_elements[item_index]
                        walk_push_to_template(fig_el, template_el, skip)
                    else:
                        template_el = fig_el.__class__()
                        walk_push_to_template(fig_el, template_el, skip)
                        template_elements.append(template_el)
                        template_element_names.append(fig_el.name)
                    fig_el.name = element_name
                else:
                    walk_push_to_template(fig_el, template_propdefaults, skip)
            template_obj[prop] = template_elements
        elif not validator.array_ok or not is_array(fig_val):
            template_obj[prop] = fig_val
            try:
                fig_obj[prop] = None
            except ValueError:
                pass