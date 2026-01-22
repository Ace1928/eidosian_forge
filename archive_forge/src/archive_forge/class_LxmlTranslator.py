from . import etree
class LxmlTranslator(external_cssselect.GenericTranslator):
    """
    A custom CSS selector to XPath translator with lxml-specific extensions.
    """

    def xpath_contains_function(self, xpath, function):
        if function.argument_types() not in (['STRING'], ['IDENT']):
            raise ExpressionError('Expected a single string or ident for :contains(), got %r' % function.arguments)
        value = function.arguments[0].value
        return xpath.add_condition('contains(__lxml_internal_css:lower-case(string(.)), %s)' % self.xpath_literal(value.lower()))