from datetime import datetime
from prov.graph import INFERRED_ELEMENT_CLASS
from prov.model import (
import pydot
def _bundle_to_dot(dot, bundle):

    def _attach_attribute_annotation(node, record):
        attributes = list(((attr_name, value) for attr_name, value in record.attributes if attr_name not in PROV_ATTRIBUTE_QNAMES))
        if not attributes:
            return
        attributes = sorted_attributes(record.get_type(), attributes)
        ann_rows = [ANNOTATION_START_ROW]
        ann_rows.extend((ANNOTATION_ROW_TEMPLATE % (attr.uri, escape(str(attr)), ' href="%s"' % value.uri if isinstance(value, Identifier) else '', escape(str(value) if not isinstance(value, datetime) else str(value.isoformat()))) for attr, value in attributes))
        ann_rows.append(ANNOTATION_END_ROW)
        count[3] += 1
        annotations = pydot.Node('ann%d' % count[3], label='\n'.join(ann_rows), **ANNOTATION_STYLE)
        dot.add_node(annotations)
        dot.add_edge(pydot.Edge(annotations, node, **ANNOTATION_LINK_STYLE))

    def _add_bundle(bundle):
        count[2] += 1
        subdot = pydot.Cluster(graph_name='c%d' % count[2], URL=f'"{bundle.identifier.uri}"')
        if use_labels:
            if bundle.label == bundle.identifier:
                bundle_label = f'"{bundle.label}"'
            else:
                bundle_label = f'<{bundle.label}<br /><font color="#333333" point-size="10">{bundle.identifier}</font>>'
            subdot.set_label(f'"{bundle_label}"')
        else:
            subdot.set_label('"%s"' % str(bundle.identifier))
        _bundle_to_dot(subdot, bundle)
        dot.add_subgraph(subdot)
        return subdot

    def _add_node(record):
        count[0] += 1
        node_id = 'n%d' % count[0]
        if use_labels:
            if record.label == record.identifier:
                node_label = f'"{record.label}"'
            else:
                node_label = f'<{record.label}<br /><font color="#333333" point-size="10">{record.identifier}</font>>'
        else:
            node_label = f'"{record.identifier}"'
        uri = record.identifier.uri
        style = DOT_PROV_STYLE[record.get_type()]
        node = pydot.Node(node_id, label=node_label, URL='"%s"' % uri, **style)
        node_map[uri] = node
        dot.add_node(node)
        if show_element_attributes:
            _attach_attribute_annotation(node, rec)
        return node

    def _add_generic_node(qname, prov_type=None):
        count[0] += 1
        node_id = 'n%d' % count[0]
        node_label = f'"{qname}"'
        uri = qname.uri
        style = GENERIC_NODE_STYLE[prov_type] if prov_type else DOT_PROV_STYLE[0]
        node = pydot.Node(node_id, label=node_label, URL='"%s"' % uri, **style)
        node_map[uri] = node
        dot.add_node(node)
        return node

    def _get_bnode():
        count[1] += 1
        bnode_id = 'b%d' % count[1]
        bnode = pydot.Node(bnode_id, label='""', shape='point', color='gray')
        dot.add_node(bnode)
        return bnode

    def _get_node(qname, prov_type=None):
        if qname is None:
            return _get_bnode()
        uri = qname.uri
        if uri not in node_map:
            _add_generic_node(qname, prov_type)
        return node_map[uri]
    records = bundle.get_records()
    relations = []
    for rec in records:
        if rec.is_element():
            _add_node(rec)
        else:
            relations.append(rec)
    if not bundle.is_bundle():
        for bundle in bundle.bundles:
            _add_bundle(bundle)
    for rec in relations:
        args = rec.args
        if not args:
            continue
        attr_names, nodes = zip(*((attr_name, value) for attr_name, value in rec.formal_attributes if attr_name in PROV_ATTRIBUTE_QNAMES))
        inferred_types = list(map(INFERRED_ELEMENT_CLASS.get, attr_names))
        other_attributes = [(attr_name, value) for attr_name, value in rec.attributes if attr_name not in PROV_ATTRIBUTE_QNAMES]
        add_attribute_annotation = show_relation_attributes and other_attributes
        add_nary_elements = len(nodes) > 2 and show_nary
        style = DOT_PROV_STYLE[rec.get_type()]
        if len(nodes) < 2:
            continue
        if add_nary_elements or add_attribute_annotation:
            bnode = _get_bnode()
            dot.add_edge(pydot.Edge(_get_node(nodes[0], inferred_types[0]), bnode, arrowhead='none', **style))
            style = dict(style)
            del style['label']
            dot.add_edge(pydot.Edge(bnode, _get_node(nodes[1], inferred_types[1]), **style))
            if add_nary_elements:
                style['color'] = 'gray'
                style['fontcolor'] = 'dimgray'
                for attr_name, node, inferred_type in zip(attr_names[2:], nodes[2:], inferred_types[2:]):
                    if node is not None:
                        style['label'] = attr_name.localpart
                        dot.add_edge(pydot.Edge(bnode, _get_node(node, inferred_type), **style))
            if add_attribute_annotation:
                _attach_attribute_annotation(bnode, rec)
        else:
            dot.add_edge(pydot.Edge(_get_node(nodes[0], inferred_types[0]), _get_node(nodes[1], inferred_types[1]), **style))