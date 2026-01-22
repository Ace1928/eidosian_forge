import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
def _add_edges(self, transitions, container):
    edges_attr = defaultdict(lambda: defaultdict(dict))
    for transition in transitions:
        src = transition['source']
        try:
            dst = transition['dest']
        except KeyError:
            dst = src
        if edges_attr[src][dst]:
            attr = edges_attr[src][dst]
            attr[attr['label_pos']] = ' | '.join([edges_attr[src][dst][attr['label_pos']], self._transition_label(transition)])
        else:
            edges_attr[src][dst] = self._create_edge_attr(src, dst, transition)
    for custom_src, dests in self.custom_styles['edge'].items():
        for custom_dst, style in dests.items():
            if style and (custom_src not in edges_attr or custom_dst not in edges_attr[custom_src]):
                edges_attr[custom_src][custom_dst] = self._create_edge_attr(custom_src, custom_dst, {'trigger': '', 'dest': ''})
    for src, dests in edges_attr.items():
        for dst, attr in dests.items():
            del attr['label_pos']
            style = self.custom_styles['edge'][src][dst]
            attr.update(**self.machine.style_attributes['edge'][style])
            container.edge(attr.pop('source'), attr.pop('dest'), **attr)