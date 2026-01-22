from itertools import chain
from nltk.internals import Counter
def _to_depgraph(self, nodes, head, rel):
    index = len(nodes)
    nodes[index].update({'address': index, 'word': self.pred[0], 'tag': self.pred[1], 'head': head, 'rel': rel})
    for feature in sorted(self):
        for item in sorted(self[feature]):
            if isinstance(item, FStructure):
                item._to_depgraph(nodes, index, feature)
            elif isinstance(item, tuple):
                new_index = len(nodes)
                nodes[new_index].update({'address': new_index, 'word': item[0], 'tag': item[1], 'head': index, 'rel': feature})
            elif isinstance(item, list):
                for n in item:
                    n._to_depgraph(nodes, index, feature)
            else:
                raise Exception('feature %s is not an FStruct, a list, or a tuple' % feature)