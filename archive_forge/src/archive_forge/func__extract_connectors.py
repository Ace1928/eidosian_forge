import collections
import enum
from taskflow.engines.action_engine import compiler as co
def _extract_connectors(execution_graph, starting_node, direction, through_flows=True, through_retries=True, through_tasks=True):
    if direction == Direction.FORWARD:
        connected_iter = execution_graph.successors
    else:
        connected_iter = execution_graph.predecessors
    connected_to_functors = {}
    if through_flows:
        connected_to_functors[co.FLOW] = connected_iter
        connected_to_functors[co.FLOW_END] = connected_iter
    if through_retries:
        connected_to_functors[co.RETRY] = connected_iter
    if through_tasks:
        connected_to_functors[co.TASK] = connected_iter
    return (connected_iter(starting_node), connected_to_functors)