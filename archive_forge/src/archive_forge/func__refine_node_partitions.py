import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@classmethod
def _refine_node_partitions(cls, graph, node_partitions, edge_colors, branch=False):
    """
        Given a partition of nodes in graph, make the partitions smaller such
        that all nodes in a partition have 1) the same color, and 2) the same
        number of edges to specific other partitions.
        """

    def equal_color(node1, node2):
        return node_edge_colors[node1] == node_edge_colors[node2]
    node_partitions = list(node_partitions)
    node_colors = partition_to_color(node_partitions)
    node_edge_colors = cls._find_node_edge_color(graph, node_colors, edge_colors)
    if all((are_all_equal((node_edge_colors[node] for node in partition)) for partition in node_partitions)):
        yield node_partitions
        return
    new_partitions = []
    output = [new_partitions]
    for partition in node_partitions:
        if not are_all_equal((node_edge_colors[node] for node in partition)):
            refined = make_partitions(partition, equal_color)
            if branch and len(refined) != 1 and (len({len(r) for r in refined}) != len([len(r) for r in refined])):
                permutations = cls._get_permutations_by_length(refined)
                new_output = []
                for n_p in output:
                    for permutation in permutations:
                        new_output.append(n_p + list(permutation[0]))
                output = new_output
            else:
                for n_p in output:
                    n_p.extend(sorted(refined, key=len))
        else:
            for n_p in output:
                n_p.append(partition)
    for n_p in output:
        yield from cls._refine_node_partitions(graph, n_p, edge_colors, branch)