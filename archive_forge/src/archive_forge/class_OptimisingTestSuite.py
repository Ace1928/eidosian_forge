import heapq
import inspect
import unittest
from pbr.version import VersionInfo
class OptimisingTestSuite(unittest.TestSuite):
    """A resource creation optimising TestSuite."""
    known_suite_classes = None

    def adsorbSuite(self, test_case_or_suite):
        """Deprecated. Use addTest instead."""
        self.addTest(test_case_or_suite)

    def addTest(self, test_case_or_suite):
        """Add `test_case_or_suite`, unwrapping standard TestSuites.

        This means that any containing unittest.TestSuites will be removed,
        while any custom test suites will be 'distributed' across their
        members. Thus addTest(CustomSuite([a, b])) will result in
        CustomSuite([a]) and CustomSuite([b]) being added to this suite.
        """
        try:
            tests = iter(test_case_or_suite)
        except TypeError:
            unittest.TestSuite.addTest(self, test_case_or_suite)
            return
        if test_case_or_suite.__class__ in self.__class__.known_suite_classes:
            for test in tests:
                self.adsorbSuite(test)
        else:
            for test in tests:
                unittest.TestSuite.addTest(self, test_case_or_suite.__class__([test]))

    def cost_of_switching(self, old_resource_set, new_resource_set):
        """Cost of switching from 'old_resource_set' to 'new_resource_set'.

        This is calculated by adding the cost of tearing down unnecessary
        resources to the cost of setting up the newly-needed resources.

        Note that resources which are always dirtied may skew the predicted
        skew the cost of switching because they are considered common, even
        when reusing them may actually be equivalent to a teardown+setup
        operation.
        """
        new_resources = new_resource_set - old_resource_set
        gone_resources = old_resource_set - new_resource_set
        return sum((resource.setUpCost for resource in new_resources)) + sum((resource.tearDownCost for resource in gone_resources))

    def switch(self, old_resource_set, new_resource_set, result):
        """Switch from 'old_resource_set' to 'new_resource_set'.

        Tear down resources in old_resource_set that aren't in
        new_resource_set and set up resources that are in new_resource_set but
        not in old_resource_set.

        :param result: TestResult object to report activity on.
        """
        new_resources = new_resource_set - old_resource_set
        old_resources = old_resource_set - new_resource_set
        for resource in old_resources:
            resource.finishedWith(resource._currentResource, result)
        for resource in new_resources:
            resource.getResource(result)

    def run(self, result):
        self.sortTests()
        current_resources = set()
        for test in self._tests:
            if result.shouldStop:
                break
            resources = getattr(test, 'resources', [])
            new_resources = set()
            for name, resource in resources:
                new_resources.update(resource.neededResources())
            self.switch(current_resources, new_resources, result)
            current_resources = new_resources
            test(result)
        self.switch(current_resources, set(), result)
        return result

    def sortTests(self):
        """Attempt to topographically sort the contained tests.

        This function biases to reusing a resource: it assumes that resetting
        a resource is usually cheaper than a teardown + setup; and that most
        resources are not dirtied by most tests.

        Feel free to override to improve the sort behaviour.
        """
        resource_set_tests = split_by_resources(self._tests)
        resource_set_graph = _resource_graph(resource_set_tests)
        no_resources = frozenset()
        partitions = _strongly_connected_components(resource_set_graph, no_resources)
        result = []
        for partition in partitions:
            if partition == [no_resources]:
                continue
            order = self._makeOrder(partition)
            for resource_set in order:
                result.extend(resource_set_tests[resource_set])
        result.extend(resource_set_tests[no_resources])
        self._tests = result

    def _getGraph(self, resource_sets):
        """Build a graph of the resource-using nodes.

        This special cases set(['root']) to be a node with no resources and
        edges to everything.

        :return: A complete directed graph of the switching costs
            between each resource combination. Note that links from N to N are
            not included.
        """
        no_resources = frozenset()
        graph = {}
        root = set(['root'])
        for from_set in resource_sets:
            graph[from_set] = {}
            if from_set == root:
                from_resources = no_resources
            else:
                from_resources = from_set
            for to_set in resource_sets:
                if from_set is to_set:
                    continue
                if to_set == root:
                    continue
                else:
                    to_resources = to_set
                graph[from_set][to_set] = self.cost_of_switching(from_resources, to_resources)
        return graph

    def _makeOrder(self, partition):
        """Return a order for the resource sets in partition."""
        root = frozenset(['root'])
        partition.add(root)
        partition.discard(frozenset())
        digraph = self._getGraph(partition)
        primes = {}
        prime = frozenset(['prime'])
        for node in digraph:
            primes[node] = node.union(prime)
        graph = _digraph_to_graph(digraph, primes)
        mst = _kruskals_graph_MST(graph)
        node = root
        cycle = [node]
        steps = 2 * (len(mst) - 1)
        for step in range(steps):
            found = False
            outgoing = None
            for outgoing in mst[node]:
                if node in mst[outgoing]:
                    del mst[node][outgoing]
                    node = outgoing
                    cycle.append(node)
                    found = True
                    break
            if not found:
                del mst[node][outgoing]
                node = outgoing
                cycle.append(node)
        visited = set()
        order = []
        for node in cycle:
            if node in visited:
                continue
            if node in primes:
                order.append(node)
            visited.add(node)
        assert order[0] == root
        return order[1:]