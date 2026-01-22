import threading
from tensorboard import errors
def GraphInfo(self, run, graph_id):
    """Get the information regarding a TensorFlow graph.

        Args:
          run: Name of the run.
          graph_id: Debugger-generated ID of the graph in question.
            This information is available in the return values
            of `GraphOpInfo`, `GraphExecution`, etc.

        Returns:
          A JSON-serializable object containing the information regarding
            the TensorFlow graph.

        Raises:
          NotFoundError if the graph_id is not known to the debugger.
        """
    runs = self.Runs()
    if run not in runs:
        return None
    try:
        graph = self._reader.graph_by_id(graph_id)
    except KeyError:
        raise errors.NotFoundError('There is no graph with ID "%s"' % graph_id)
    return graph.to_json()