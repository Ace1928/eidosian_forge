import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
def _parameterized_repr_html(p, open):
    """HTML representation for a Parameterized object"""
    if isinstance(p, Parameterized):
        cls = p.__class__
        title = cls.name + '()'
        value_field = 'Value'
    else:
        cls = p
        title = cls.name
        value_field = 'Default'
    tooltip_css = '\n.param-doc-tooltip{\n  position: relative;\n  cursor: help;\n}\n.param-doc-tooltip:hover:after{\n  content: attr(data-tooltip);\n  background-color: black;\n  color: #fff;\n  border-radius: 3px;\n  padding: 10px;\n  position: absolute;\n  z-index: 1;\n  top: -5px;\n  left: 100%;\n  margin-left: 10px;\n  min-width: 250px;\n}\n.param-doc-tooltip:hover:before {\n  content: "";\n  position: absolute;\n  top: 50%;\n  left: 100%;\n  margin-top: -5px;\n  border-width: 5px;\n  border-style: solid;\n  border-color: transparent black transparent transparent;\n}\n'
    openstr = ' open' if open else ''
    param_values = p.param.values().items()
    contents = ''.join((_get_param_repr(key, val, p.param[key]) for key, val in param_values))
    return f'<style>{tooltip_css}</style>\n<details {openstr}>\n <summary style="display:list-item; outline:none;">\n  <tt>{title}</tt>\n </summary>\n <div style="padding-left:10px; padding-bottom:5px;">\n  <table style="max-width:100%; border:1px solid #AAAAAA;">\n   <tr><th style="text-align:left;">Name</th><th style="text-align:left;">{value_field}</th><th style="text-align:left;">Type</th><th>Range</th></tr>\n{contents}\n  </table>\n </div>\n</details>\n'