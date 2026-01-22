from .state import ExecutionContext
from .property import ProcessProperty
from .embeddedRDF import handle_embeddedRDF
from .host import HostLanguage, host_dom_transforms
from rdflib import URIRef
from rdflib import BNode
from rdflib import RDF as ns_rdf
from . import IncorrectBlankNodeUsage, err_no_blank_node
from .utils import has_one_of_attributes
def _parse_1_0(node, graph, parent_object, incoming_state, parent_incomplete_triples):
    """The (recursive) step of handling a single node. See the
    U{RDFa 1.0 syntax document<http://www.w3.org/TR/rdfa-syntax>} for further details.
    
    This is the RDFa 1.0 version.

    @param node: the DOM node to handle
    @param graph: the RDF graph
    @type graph: RDFLib's Graph object instance
    @param parent_object: the parent's object, as an RDFLib URIRef
    @param incoming_state: the inherited state (namespaces, lang, etc.)
    @type incoming_state: L{state.ExecutionContext}
    @param parent_incomplete_triples: list of hanging triples (the missing resource set to None) to be handled (or not)
    by the current node.
    @return: whether the caller has to complete it's parent's incomplete triples
    @rtype: Boolean
    """
    state = ExecutionContext(node, graph, inherited_state=incoming_state)
    handle_role_attribute(node, graph, state)
    if state.options.embedded_rdf and node.nodeType == node.ELEMENT_NODE and handle_embeddedRDF(node, graph, state):
        return
    if state.options.host_language in host_dom_transforms and node.nodeType == node.ELEMENT_NODE:
        for func in host_dom_transforms[state.options.host_language]:
            func(node, state)
    if not has_one_of_attributes(node, 'href', 'resource', 'about', 'property', 'rel', 'rev', 'typeof', 'src'):
        for n in node.childNodes:
            if n.nodeType == node.ELEMENT_NODE:
                parse_one_node(n, graph, parent_object, state, parent_incomplete_triples)
        return
    current_subject = None
    current_object = None
    if has_one_of_attributes(node, 'rel', 'rev'):
        current_subject = state.getResource('about', 'src')
        if current_subject == None:
            if node.hasAttribute('typeof'):
                current_subject = BNode()
            else:
                current_subject = parent_object
        else:
            state.reset_list_mapping(origin=current_subject)
        current_object = state.getResource('resource', 'href')
    else:
        current_subject = state.getResource('about', 'src', 'resource', 'href')
        if current_subject == None:
            if node.hasAttribute('typeof'):
                current_subject = BNode()
            else:
                current_subject = parent_object
            current_subject = parent_object
        else:
            state.reset_list_mapping(origin=current_subject)
        current_object = current_subject
    for defined_type in state.getURI('typeof'):
        graph.add((current_subject, ns_rdf['type'], defined_type))
    incomplete_triples = []
    for prop in state.getURI('rel'):
        if not isinstance(prop, BNode):
            theTriple = (current_subject, prop, current_object)
            if current_object != None:
                graph.add(theTriple)
            else:
                incomplete_triples.append(theTriple)
        else:
            state.options.add_warning(err_no_blank_node % 'rel', warning_type=IncorrectBlankNodeUsage, node=node.nodeName)
    for prop in state.getURI('rev'):
        if not isinstance(prop, BNode):
            theTriple = (current_object, prop, current_subject)
            if current_object != None:
                graph.add(theTriple)
            else:
                incomplete_triples.append(theTriple)
        else:
            state.options.add_warning(err_no_blank_node % 'rev', warning_type=IncorrectBlankNodeUsage, node=node.nodeName)
    if node.hasAttribute('property'):
        ProcessProperty(node, graph, current_subject, state).generate_1_0()
    if current_object == None:
        object_to_children = BNode()
    else:
        object_to_children = current_object
    for n in node.childNodes:
        if n.nodeType == node.ELEMENT_NODE:
            _parse_1_0(n, graph, object_to_children, state, incomplete_triples)
    for s, p, o in parent_incomplete_triples:
        if s == None and o == None:
            incoming_state.add_to_list_mapping(p, current_subject)
        else:
            if s == None:
                s = current_subject
            if o == None:
                o = current_subject
            graph.add((s, p, o))
    return