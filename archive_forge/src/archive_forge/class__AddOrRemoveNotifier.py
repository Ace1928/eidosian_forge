class _AddOrRemoveNotifier:
    """ Callable for adding or removing notifiers.

    See ``add_or_remove_notifiers`` for the input parameters.
    """

    def __init__(self, *, object, graph, handler, target, dispatcher, remove):
        self.object = object
        self.graph = graph
        self.handler = handler
        self.target = target
        self.dispatcher = dispatcher
        self.remove = remove
        self._processed = []

    def __call__(self):
        """ Main function for adding/removing notifiers.
        """
        steps = [self._add_or_remove_notifiers, self._add_or_remove_maintainers, self._add_or_remove_children_notifiers, self._add_or_remove_extra_graphs]
        if self.remove:
            steps = steps[::-1]
        try:
            for step in steps:
                step()
        except Exception:
            while self._processed:
                notifier, observable = self._processed.pop()
                if self.remove:
                    notifier.add_to(observable)
                else:
                    notifier.remove_from(observable)
            raise
        else:
            self._processed.clear()

    def _add_or_remove_extra_graphs(self):
        """ Add or remove additional ObserverGraph contributed by the root
        observer. e.g. for handing trait_added event.
        """
        for extra_graph in self.graph.node.iter_extra_graphs(self.graph):
            add_or_remove_notifiers(object=self.object, graph=extra_graph, handler=self.handler, target=self.target, dispatcher=self.dispatcher, remove=self.remove)

    def _add_or_remove_children_notifiers(self):
        """ Recursively add or remove notifiers for the children ObserverGraph.
        """
        for child_graph in self.graph.children:
            for next_object in self.graph.node.iter_objects(self.object):
                add_or_remove_notifiers(object=next_object, graph=child_graph, handler=self.handler, target=self.target, dispatcher=self.dispatcher, remove=self.remove)

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

    def _add_or_remove_notifiers(self):
        """ Add or remove user notifiers for the objects observed by the root
        observer.
        """
        if not self.graph.node.notify:
            return
        for observable in self.graph.node.iter_observables(self.object):
            notifier = self.graph.node.get_notifier(handler=self.handler, target=self.target, dispatcher=self.dispatcher)
            if self.remove:
                notifier.remove_from(observable)
            else:
                notifier.add_to(observable)
            self._processed.append((notifier, observable))