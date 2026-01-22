from .links_base import Strand, Crossing, Link
import random
import collections
class Frontier(BiDict):

    def overlap_indices(self, crossing):
        neighbors = [cs.opposite() for cs in crossing.crossing_strands()]
        return [self[ns] for ns in neighbors if ns in self]

    def overlap_is_consecutive(self, crossing):
        overlap = self.overlap_indices(crossing)
        return len(overlap) > 0 and is_range(overlap)

    def biggest_all_consecutive_overlap(self):
        """
        Return a random crossing from among those with the maximal possible
        overlap.
        """
        overlap_indices = collections.defaultdict(list)
        for i, cs in self.int_to_set.items():
            overlap_indices[cs.opposite()[0]].append(i)
        possible = []
        for crossing, overlap in overlap_indices.items():
            overlap = sorted(overlap)
            if is_range(overlap):
                possible.append((len(overlap), min(overlap), crossing))
        max_overlap = max(possible)[0]
        good_choices = [pos for pos in possible if pos[0] == max_overlap]
        return random.choice(good_choices)