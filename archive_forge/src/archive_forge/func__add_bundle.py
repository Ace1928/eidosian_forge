from datetime import datetime
from prov.graph import INFERRED_ELEMENT_CLASS
from prov.model import (
import pydot
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