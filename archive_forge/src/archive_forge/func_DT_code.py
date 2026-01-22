import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def DT_code(self, DT_alpha=False, flips=False):
    """
        The Dowker-Thistlethwaite code for the link in either numerical or
        alphabetical form.

        >>> L = Link('K8n1')
        >>> L.DT_code(DT_alpha=True, flips=True)
        'DT[hahCHeAgbdf.11101000]'

        In the alphabetical form, the first letter determines the
        number C of crossings, the second the number L of link
        components, and the next L gives the number of crossings on
        each component; subsequent letters describe each crossing with
        'a' being 2, 'A' being -2, etc.
        """
    DT_info = [c.DT_info() for c in self.crossings]
    first_to_second = {first: second for first, second, _ in DT_info}
    first_to_flip = {first: flip for first, _, flip in DT_info}
    odd_labels = enumerate_lists(self.link_components, n=1, filter=lambda x: x % 2 == 1)
    DT = [tuple((first_to_second[x] for x in component)) for component in odd_labels]
    the_flips = (first_to_flip[x] for x in sum(odd_labels, []))
    if DT_alpha:
        if len(self) > 52:
            raise ValueError('Too many crossing for alphabetic DT code')
        DT_alphabet = '_abcdefghijklmnopqrstuvwxyzZYXWVUTSRQPONMLKJIHGFEDCBA'
        init_data = [len(self), len(DT)] + [len(c) for c in DT]
        DT = ''.join([DT_alphabet[x] for x in init_data] + [DT_alphabet[x >> 1] for x in sum(DT, tuple())])
        if flips:
            DT += '.' + ''.join((repr(flip) for flip in the_flips))
        return f'DT[{DT}]'
    elif flips:
        return (DT, list(the_flips))
    else:
        return DT