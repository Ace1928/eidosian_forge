def _add_or_remove_maintainers(self):
    """ Add or remove notifiers for maintaining children notifiers when
        the objects being observed by the root observer change.
        """
    for observable in self.graph.node.iter_observables(self.object):
        for child_graph in self.graph.children:
            change_notifier = self.graph.node.get_maintainer(graph=child_graph, handler=self.handler, target=self.target, dispatcher=self.dispatcher)
            if self.remove:
                change_notifier.remove_from(observable)
            else:
                change_notifier.add_to(observable)
            self._processed.append((change_notifier, observable))