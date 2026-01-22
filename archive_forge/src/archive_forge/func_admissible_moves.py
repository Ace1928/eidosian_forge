from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def admissible_moves(link):
    circles = seifert_circles(link)
    cs_to_seifert_circle = {}
    pairs = []
    seifert_circle_pairs = []
    for cs in link.crossing_strands():
        cep = seifert_crossing_entry(cs)
        for c in circles:
            if cep in c:
                break
        cs_to_seifert_circle[cs] = circles.index(c)
    for face in link.faces():
        for cs1, cs2 in combinations(face, 2):
            circle1, circle2 = (cs_to_seifert_circle[cs1], cs_to_seifert_circle[cs2])
            if circle1 != circle2:
                pairs.append((cs1, cs2))
                seifert_circle_pairs.append((circle1, circle2))
    return (pairs, seifert_circle_pairs)