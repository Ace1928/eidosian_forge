import re
import typing
import warnings
from typing import Optional
from cssselect.parser import (
class GenericTranslator:
    """
    Translator for "generic" XML documents.

    Everything is case-sensitive, no assumption is made on the meaning
    of element names and attribute names.

    """
    combinator_mapping = {' ': 'descendant', '>': 'child', '+': 'direct_adjacent', '~': 'indirect_adjacent'}
    attribute_operator_mapping = {'exists': 'exists', '=': 'equals', '~=': 'includes', '|=': 'dashmatch', '^=': 'prefixmatch', '$=': 'suffixmatch', '*=': 'substringmatch', '!=': 'different'}
    id_attribute = 'id'
    lang_attribute = 'xml:lang'
    lower_case_element_names = False
    lower_case_attribute_names = False
    lower_case_attribute_values = False
    xpathexpr_cls = XPathExpr

    def css_to_xpath(self, css: str, prefix: str='descendant-or-self::') -> str:
        """Translate a *group of selectors* to XPath.

        Pseudo-elements are not supported here since XPath only knows
        about "real" elements.

        :param css:
            A *group of selectors* as a string.
        :param prefix:
            This string is prepended to the XPath expression for each selector.
            The default makes selectors scoped to the context node’s subtree.
        :raises:
            :class:`~cssselect.SelectorSyntaxError` on invalid selectors,
            :class:`ExpressionError` on unknown/unsupported selectors,
            including pseudo-elements.
        :returns:
            The equivalent XPath 1.0 expression as a string.

        """
        return ' | '.join((self.selector_to_xpath(selector, prefix, translate_pseudo_elements=True) for selector in parse(css)))

    def selector_to_xpath(self, selector: Selector, prefix: str='descendant-or-self::', translate_pseudo_elements: bool=False) -> str:
        """Translate a parsed selector to XPath.


        :param selector:
            A parsed :class:`Selector` object.
        :param prefix:
            This string is prepended to the resulting XPath expression.
            The default makes selectors scoped to the context node’s subtree.
        :param translate_pseudo_elements:
            Unless this is set to ``True`` (as :meth:`css_to_xpath` does),
            the :attr:`~Selector.pseudo_element` attribute of the selector
            is ignored.
            It is the caller's responsibility to reject selectors
            with pseudo-elements, or to account for them somehow.
        :raises:
            :class:`ExpressionError` on unknown/unsupported selectors.
        :returns:
            The equivalent XPath 1.0 expression as a string.

        """
        tree = getattr(selector, 'parsed_tree', None)
        if not tree:
            raise TypeError('Expected a parsed selector, got %r' % (selector,))
        xpath = self.xpath(tree)
        assert isinstance(xpath, self.xpathexpr_cls)
        if translate_pseudo_elements and selector.pseudo_element:
            xpath = self.xpath_pseudo_element(xpath, selector.pseudo_element)
        return (prefix or '') + str(xpath)

    def xpath_pseudo_element(self, xpath: XPathExpr, pseudo_element: PseudoElement) -> XPathExpr:
        """Translate a pseudo-element.

        Defaults to not supporting pseudo-elements at all,
        but can be overridden by sub-classes.

        """
        raise ExpressionError('Pseudo-elements are not supported.')

    @staticmethod
    def xpath_literal(s: str) -> str:
        s = str(s)
        if "'" not in s:
            s = "'%s'" % s
        elif '"' not in s:
            s = '"%s"' % s
        else:
            s = 'concat(%s)' % ','.join([("'" in part and '"%s"' or "'%s'") % part for part in split_at_single_quotes(s) if part])
        return s

    def xpath(self, parsed_selector: Tree) -> XPathExpr:
        """Translate any parsed selector object."""
        type_name = type(parsed_selector).__name__
        method = getattr(self, 'xpath_%s' % type_name.lower(), None)
        if method is None:
            raise ExpressionError('%s is not supported.' % type_name)
        return typing.cast(XPathExpr, method(parsed_selector))

    def xpath_combinedselector(self, combined: CombinedSelector) -> XPathExpr:
        """Translate a combined selector."""
        combinator = self.combinator_mapping[combined.combinator]
        method = getattr(self, 'xpath_%s_combinator' % combinator)
        return typing.cast(XPathExpr, method(self.xpath(combined.selector), self.xpath(combined.subselector)))

    def xpath_negation(self, negation: Negation) -> XPathExpr:
        xpath = self.xpath(negation.selector)
        sub_xpath = self.xpath(negation.subselector)
        sub_xpath.add_name_test()
        if sub_xpath.condition:
            return xpath.add_condition('not(%s)' % sub_xpath.condition)
        else:
            return xpath.add_condition('0')

    def xpath_relation(self, relation: Relation) -> XPathExpr:
        xpath = self.xpath(relation.selector)
        combinator = relation.combinator
        subselector = relation.subselector
        right = self.xpath(subselector.parsed_tree)
        method = getattr(self, 'xpath_relation_%s_combinator' % self.combinator_mapping[typing.cast(str, combinator.value)])
        return typing.cast(XPathExpr, method(xpath, right))

    def xpath_matching(self, matching: Matching) -> XPathExpr:
        xpath = self.xpath(matching.selector)
        exprs = [self.xpath(selector) for selector in matching.selector_list]
        for e in exprs:
            e.add_name_test()
            if e.condition:
                xpath.add_condition(e.condition, 'or')
        return xpath

    def xpath_specificityadjustment(self, matching: SpecificityAdjustment) -> XPathExpr:
        xpath = self.xpath(matching.selector)
        exprs = [self.xpath(selector) for selector in matching.selector_list]
        for e in exprs:
            e.add_name_test()
            if e.condition:
                xpath.add_condition(e.condition, 'or')
        return xpath

    def xpath_function(self, function: Function) -> XPathExpr:
        """Translate a functional pseudo-class."""
        method_name = 'xpath_%s_function' % function.name.replace('-', '_')
        method = getattr(self, method_name, None)
        if not method:
            raise ExpressionError('The pseudo-class :%s() is unknown' % function.name)
        return typing.cast(XPathExpr, method(self.xpath(function.selector), function))

    def xpath_pseudo(self, pseudo: Pseudo) -> XPathExpr:
        """Translate a pseudo-class."""
        method_name = 'xpath_%s_pseudo' % pseudo.ident.replace('-', '_')
        method = getattr(self, method_name, None)
        if not method:
            raise ExpressionError('The pseudo-class :%s is unknown' % pseudo.ident)
        return typing.cast(XPathExpr, method(self.xpath(pseudo.selector)))

    def xpath_attrib(self, selector: Attrib) -> XPathExpr:
        """Translate an attribute selector."""
        operator = self.attribute_operator_mapping[selector.operator]
        method = getattr(self, 'xpath_attrib_%s' % operator)
        if self.lower_case_attribute_names:
            name = selector.attrib.lower()
        else:
            name = selector.attrib
        safe = is_safe_name(name)
        if selector.namespace:
            name = '%s:%s' % (selector.namespace, name)
            safe = safe and is_safe_name(selector.namespace)
        if safe:
            attrib = '@' + name
        else:
            attrib = 'attribute::*[name() = %s]' % self.xpath_literal(name)
        if selector.value is None:
            value = None
        elif self.lower_case_attribute_values:
            value = typing.cast(str, selector.value.value).lower()
        else:
            value = selector.value.value
        return typing.cast(XPathExpr, method(self.xpath(selector.selector), attrib, value))

    def xpath_class(self, class_selector: Class) -> XPathExpr:
        """Translate a class selector."""
        xpath = self.xpath(class_selector.selector)
        return self.xpath_attrib_includes(xpath, '@class', class_selector.class_name)

    def xpath_hash(self, id_selector: Hash) -> XPathExpr:
        """Translate an ID selector."""
        xpath = self.xpath(id_selector.selector)
        return self.xpath_attrib_equals(xpath, '@id', id_selector.id)

    def xpath_element(self, selector: Element) -> XPathExpr:
        """Translate a type or universal selector."""
        element = selector.element
        if not element:
            element = '*'
            safe = True
        else:
            safe = bool(is_safe_name(element))
            if self.lower_case_element_names:
                element = element.lower()
        if selector.namespace:
            element = '%s:%s' % (selector.namespace, element)
            safe = safe and bool(is_safe_name(selector.namespace))
        xpath = self.xpathexpr_cls(element=element)
        if not safe:
            xpath.add_name_test()
        return xpath

    def xpath_descendant_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a child, grand-child or further descendant of left"""
        return left.join('/descendant-or-self::*/', right)

    def xpath_child_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is an immediate child of left"""
        return left.join('/', right)

    def xpath_direct_adjacent_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a sibling immediately after left"""
        xpath = left.join('/following-sibling::', right)
        xpath.add_name_test()
        return xpath.add_condition('position() = 1')

    def xpath_indirect_adjacent_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a sibling after left, immediately or not"""
        return left.join('/following-sibling::', right)

    def xpath_relation_descendant_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a child, grand-child or further descendant of left; select left"""
        return left.join('[descendant::', right, closing_combiner=']', has_inner_condition=True)

    def xpath_relation_child_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is an immediate child of left; select left"""
        return left.join('[./', right, closing_combiner=']')

    def xpath_relation_direct_adjacent_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a sibling immediately after left; select left"""
        xpath = left.add_condition("following-sibling::*[(name() = '{}') and (position() = 1)]".format(right.element))
        return xpath

    def xpath_relation_indirect_adjacent_combinator(self, left: XPathExpr, right: XPathExpr) -> XPathExpr:
        """right is a sibling after left, immediately or not; select left"""
        return left.join('[following-sibling::', right, closing_combiner=']')

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

    def xpath_nth_last_child_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        return self.xpath_nth_child_function(xpath, function, last=True)

    def xpath_nth_of_type_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        if xpath.element == '*':
            raise ExpressionError('*:nth-of-type() is not implemented')
        return self.xpath_nth_child_function(xpath, function, add_name_test=False)

    def xpath_nth_last_of_type_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        if xpath.element == '*':
            raise ExpressionError('*:nth-of-type() is not implemented')
        return self.xpath_nth_child_function(xpath, function, last=True, add_name_test=False)

    def xpath_contains_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        if function.argument_types() not in (['STRING'], ['IDENT']):
            raise ExpressionError('Expected a single string or ident for :contains(), got %r' % function.arguments)
        value = typing.cast(str, function.arguments[0].value)
        return xpath.add_condition('contains(., %s)' % self.xpath_literal(value))

    def xpath_lang_function(self, xpath: XPathExpr, function: Function) -> XPathExpr:
        if function.argument_types() not in (['STRING'], ['IDENT']):
            raise ExpressionError('Expected a single string or ident for :lang(), got %r' % function.arguments)
        value = typing.cast(str, function.arguments[0].value)
        return xpath.add_condition('lang(%s)' % self.xpath_literal(value))

    def xpath_root_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition('not(parent::*)')

    def xpath_scope_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition('1')

    def xpath_first_child_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition('count(preceding-sibling::*) = 0')

    def xpath_last_child_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition('count(following-sibling::*) = 0')

    def xpath_first_of_type_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        if xpath.element == '*':
            raise ExpressionError('*:first-of-type is not implemented')
        return xpath.add_condition('count(preceding-sibling::%s) = 0' % xpath.element)

    def xpath_last_of_type_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        if xpath.element == '*':
            raise ExpressionError('*:last-of-type is not implemented')
        return xpath.add_condition('count(following-sibling::%s) = 0' % xpath.element)

    def xpath_only_child_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition('count(parent::*/child::*) = 1')

    def xpath_only_of_type_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        if xpath.element == '*':
            raise ExpressionError('*:only-of-type is not implemented')
        return xpath.add_condition('count(parent::*/child::%s) = 1' % xpath.element)

    def xpath_empty_pseudo(self, xpath: XPathExpr) -> XPathExpr:
        return xpath.add_condition('not(*) and not(string-length())')

    def pseudo_never_matches(self, xpath: XPathExpr) -> XPathExpr:
        """Common implementation for pseudo-classes that never match."""
        return xpath.add_condition('0')
    xpath_link_pseudo = pseudo_never_matches
    xpath_visited_pseudo = pseudo_never_matches
    xpath_hover_pseudo = pseudo_never_matches
    xpath_active_pseudo = pseudo_never_matches
    xpath_focus_pseudo = pseudo_never_matches
    xpath_target_pseudo = pseudo_never_matches
    xpath_enabled_pseudo = pseudo_never_matches
    xpath_disabled_pseudo = pseudo_never_matches
    xpath_checked_pseudo = pseudo_never_matches

    def xpath_attrib_exists(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        assert not value
        xpath.add_condition(name)
        return xpath

    def xpath_attrib_equals(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        assert value is not None
        xpath.add_condition('%s = %s' % (name, self.xpath_literal(value)))
        return xpath

    def xpath_attrib_different(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        assert value is not None
        if value:
            xpath.add_condition('not(%s) or %s != %s' % (name, name, self.xpath_literal(value)))
        else:
            xpath.add_condition('%s != %s' % (name, self.xpath_literal(value)))
        return xpath

    def xpath_attrib_includes(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        if value and is_non_whitespace(value):
            xpath.add_condition("%s and contains(concat(' ', normalize-space(%s), ' '), %s)" % (name, name, self.xpath_literal(' ' + value + ' ')))
        else:
            xpath.add_condition('0')
        return xpath

    def xpath_attrib_dashmatch(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        assert value is not None
        xpath.add_condition('%s and (%s = %s or starts-with(%s, %s))' % (name, name, self.xpath_literal(value), name, self.xpath_literal(value + '-')))
        return xpath

    def xpath_attrib_prefixmatch(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        if value:
            xpath.add_condition('%s and starts-with(%s, %s)' % (name, name, self.xpath_literal(value)))
        else:
            xpath.add_condition('0')
        return xpath

    def xpath_attrib_suffixmatch(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        if value:
            xpath.add_condition('%s and substring(%s, string-length(%s)-%s) = %s' % (name, name, name, len(value) - 1, self.xpath_literal(value)))
        else:
            xpath.add_condition('0')
        return xpath

    def xpath_attrib_substringmatch(self, xpath: XPathExpr, name: str, value: Optional[str]) -> XPathExpr:
        if value:
            xpath.add_condition('%s and contains(%s, %s)' % (name, name, self.xpath_literal(value)))
        else:
            xpath.add_condition('0')
        return xpath