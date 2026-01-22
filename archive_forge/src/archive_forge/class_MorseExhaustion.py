from .links_base import Strand, Crossing, Link
import random
import collections
class MorseExhaustion:
    """
    An exhaustion of a link where crossings are added in one-by-one
    so that the resulting tangle is connected at every stage.

    Starting at the given crossing, it uses a greedy algorithm to try
    to minimize the sizes of the frontiers of the intermediate tangles.

    The answer is returned as a sequence of events describing a
    MorseEncoding.

    If no initial crossing is specified, one is chosen at random.

    >>> L = Link('L2a1')
    >>> mexhaust = MorseExhaustion(L, L.crossings[0])
    >>> mexhaust
    [('cup', 0, 1), ('cup', 0, 1), ('cross', 1, 2), ('cross', 1, 2), ('cap', 0, 1), ('cap', 0, 1)]
    >>> me = MorseEncoding(mexhaust)
    >>> me.link().exterior().fundamental_group().relators() #doctest: +SNAPPY
    ['abAB']

    >>> K = Link([[0, 0, 1, 1]])  # Unknot
    >>> MorseExhaustion(K)
    [('cup', 0, 1), ('cup', 0, 1), ('cross', 1, 2), ('cap', 0, 1), ('cap', 0, 1)]
    """

    def __init__(self, link, crossing=None):
        events = []
        if link.crossings:
            if crossing is None:
                crossing = random.choice(link.crossings)
            crossings = [crossing]
            events = [('cup', 0, 1), ('cup', 0, 1), ('cross', 1, 2)]
            css = crossing.crossing_strands()
            frontier = Frontier({0: css[3], 1: css[2], 2: css[1], 3: css[0]})
            frontier_lengths = [4]
            if len(link.crossings) == 1:
                events += [('cap', 0, 1), ('cap', 0, 1)]
        else:
            crossings = []
            frontier_lengths = []
        while len(crossings) < len(link.crossings):
            overlap, i, C = frontier.biggest_all_consecutive_overlap()
            cs = frontier[i]
            cs_opp = cs.opposite()
            assert C not in crossings
            crossings.append(C)
            if overlap == 1:
                i = frontier[cs]
                events.append(('cup', i + 1, i + 2))
                if cs_opp.strand_index in {1, 3}:
                    events.append(('cross', i, i + 1))
                else:
                    events.append(('cross', i + 1, i))
                frontier.insert_space(i + 1, 2)
                for s in range(3):
                    frontier[i + s] = cs_opp.rotate(-(s + 1))
            elif overlap == 2:
                if cs_opp.strand_index in {1, 3}:
                    events.append(('cross', i, i + 1))
                else:
                    events.append(('cross', i + 1, i))
                for s in range(2):
                    frontier[i + s] = cs_opp.rotate(-(s + 1))
            elif overlap == 3:
                if cs_opp.strand_index in {1, 3}:
                    events.append(('cross', i, i + 1))
                else:
                    events.append(('cross', i + 1, i))
                events.append(('cap', i + 1, i + 2))
                frontier.pop(i + 2)
                frontier.pop(i)
                frontier[i] = cs_opp.rotate(-1)
            else:
                assert overlap == 4
                if cs_opp.rotate().strand_index in {1, 3}:
                    events.append(('cross', i + 1, i + 2))
                else:
                    events.append(('cross', i + 2, i + 1))
                events += 2 * [('cap', i, i + 1)]
                for a in range(4):
                    frontier.pop(i)
            assert frontier_lengths[-1] + 4 - 2 * overlap == len(frontier)
            assert frontier._check()
            frontier_lengths.append(len(frontier))
        c = link.unlinked_unknot_components
        events += c * [('cup', 0, 1), ('cap', 0, 1)]
        frontier_lengths += c * [2, 0]
        self.link = link
        self.crossings = crossings
        self.frontier_lengths = frontier_lengths
        self.events = events
        self.width = max(frontier_lengths) // 2

    def __repr__(self):
        return repr(self.events)

    def __iter__(self):
        return self.events.__iter__()