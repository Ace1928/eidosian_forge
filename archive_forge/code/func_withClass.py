import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
def withClass(classname, namespace=''):
    """
    Simplified version of C{L{withAttribute}} when matching on a div class - made
    difficult because C{class} is a reserved word in Python.

    Example::
        html = '''
            <div>
            Some text
            <div class="grid">1 4 0 1 0</div>
            <div class="graph">1,3 2,3 1,1</div>
            <div>this &lt;div&gt; has no class</div>
            </div>
                
        '''
        div,div_end = makeHTMLTags("div")
        div_grid = div().setParseAction(withClass("grid"))
        
        grid_expr = div_grid + SkipTo(div | div_end)("body")
        for grid_header in grid_expr.searchString(html):
            print(grid_header.body)
        
        div_any_type = div().setParseAction(withClass(withAttribute.ANY_VALUE))
        div_expr = div_any_type + SkipTo(div | div_end)("body")
        for div_header in div_expr.searchString(html):
            print(div_header.body)
    prints::
        1 4 0 1 0

        1 4 0 1 0
        1,3 2,3 1,1
    """
    classattr = '%s:class' % namespace if namespace else 'class'
    return withAttribute(**{classattr: classname})