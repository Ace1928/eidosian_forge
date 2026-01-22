from datetime import datetime
from prov.graph import INFERRED_ELEMENT_CLASS
from prov.model import (
import pydot
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