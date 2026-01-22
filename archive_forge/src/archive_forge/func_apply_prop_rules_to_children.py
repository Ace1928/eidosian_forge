import codecs
import copy
import json
import os
from urllib.parse import unquote
import bs4
from . import mf2_classes
from .dom_helpers import get_children
from .mf_helpers import unordered_list
def apply_prop_rules_to_children(parent, rules):
    for child in get_children(parent):
        classes = child.get('class', [])[:]
        child['class'] = [cl for cl in classes if not mf2_classes.is_property_class(cl)]
        for rule in rules:
            rule(child)
        if not (mf2_classes.root(classes, filtered_roots) or root(classes)):
            apply_prop_rules_to_children(child, rules)