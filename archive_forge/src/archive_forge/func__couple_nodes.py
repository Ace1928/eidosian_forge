import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def _couple_nodes(self, top_partitions, bottom_partitions, pair_idx, t_node, b_node, graph, edge_colors):
    """
        Generate new partitions from top and bottom_partitions where t_node is
        coupled to b_node. pair_idx is the index of the partitions where t_ and
        b_node can be found.
        """
    t_partition = top_partitions[pair_idx]
    b_partition = bottom_partitions[pair_idx]
    assert t_node in t_partition and b_node in b_partition
    new_top_partitions = [top.copy() for top in top_partitions]
    new_bottom_partitions = [bot.copy() for bot in bottom_partitions]
    new_t_groups = ({t_node}, t_partition - {t_node})
    new_b_groups = ({b_node}, b_partition - {b_node})
    del new_top_partitions[pair_idx]
    del new_bottom_partitions[pair_idx]
    new_top_partitions[pair_idx:pair_idx] = new_t_groups
    new_bottom_partitions[pair_idx:pair_idx] = new_b_groups
    new_top_partitions = self._refine_node_partitions(graph, new_top_partitions, edge_colors)
    new_bottom_partitions = self._refine_node_partitions(graph, new_bottom_partitions, edge_colors, branch=True)
    new_top_partitions = list(new_top_partitions)
    assert len(new_top_partitions) == 1
    new_top_partitions = new_top_partitions[0]
    for bot in new_bottom_partitions:
        yield (list(new_top_partitions), bot)