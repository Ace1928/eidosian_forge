def vocab_for_role(node, options, state):
    """
    The value of the @role attribute (defined separately in the U{Role Attribute Specification Lite<http://www.w3.org/TR/role-attribute/#using-role-in-conjunction-with-rdfa>}) should be as if a @vocab value to the
    XHTML vocabulary was defined for it. This method turns all terms in role attributes into full URI-s, so that
    this would not be an issue for the run-time.
    
    @param node: a DOM node for the top level element
    @param options: invocation options
    @type options: L{Options<pyRdfa.options>}
    @param state: top level execution state
    @type state: L{State<pyRdfa.state>}
    """
    from ..termorcurie import termname, XHTML_URI

    def handle_role(node):
        if node.hasAttribute('role'):
            old_values = node.getAttribute('role').strip().split()
            new_values = ''
            for val in old_values:
                if termname.match(val):
                    new_values += XHTML_URI + val + ' '
                else:
                    new_values += val + ' '
            node.setAttribute('role', new_values.strip())
    handle_role(node)
    for n in node.childNodes:
        if n.nodeType == node.ELEMENT_NODE:
            vocab_for_role(n, options, state)