import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
def xpath_nth_child_function(self, xpath: XPathExpr, function: Function, last: bool=False, add_name_test: bool=True) -> XPathExpr:
    try:
        a, b = parse_series(function.arguments)
    except ValueError:
        raise ExpressionError("Invalid series: '%r'" % function.arguments)
    b_min_1 = b - 1
    if a == 1 and b_min_1 <= 0:
        return xpath
    if a < 0 and b_min_1 < 0:
        return xpath.add_condition('0')
    if add_name_test:
        nodetest = '*'
    else:
        nodetest = '%s' % xpath.element
    if not last:
        siblings_count = 'count(preceding-sibling::%s)' % nodetest
    else:
        siblings_count = 'count(following-sibling::%s)' % nodetest
    if a == 0:
        return xpath.add_condition('%s = %s' % (siblings_count, b_min_1))
    expressions = []
    if a > 0:
        if b_min_1 > 0:
            expressions.append('%s >= %s' % (siblings_count, b_min_1))
    else:
        expressions.append('%s <= %s' % (siblings_count, b_min_1))
    if abs(a) != 1:
        left = siblings_count
        b_neg = -b_min_1 % abs(a)
        if b_neg != 0:
            b_neg_as_str = '+%s' % b_neg
            left = '(%s %s)' % (left, b_neg_as_str)
        expressions.append('%s mod %s = 0' % (left, a))
    if len(expressions) > 1:
        template = '(%s)'
    else:
        template = '%s'
    xpath.add_condition(' and '.join((template % expression for expression in expressions)))
    return xpath