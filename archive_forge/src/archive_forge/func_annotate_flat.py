from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def annotate_flat(self, key):
    """Determine the single-best-revision to source for each line.

        This is meant as a compatibility thunk to how annotate() used to work.
        :return: [(ann_key, line)]
            A list of tuples with a single annotation key for each line.
        """
    from .annotate import _break_annotation_tie
    custom_tiebreaker = _break_annotation_tie
    annotations, lines = self.annotate(key)
    out = []
    heads = self._get_heads_provider().heads
    append = out.append
    for annotation, line in zip(annotations, lines):
        if len(annotation) == 1:
            head = annotation[0]
        else:
            the_heads = heads(annotation)
            if len(the_heads) == 1:
                for head in the_heads:
                    break
            else:
                head = self._resolve_annotation_tie(the_heads, line, custom_tiebreaker)
        append((head, line))
    return out