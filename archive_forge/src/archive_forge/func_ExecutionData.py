import threading
from tensorboard import errors
def ExecutionData(self, run, begin, end):
    """Get Execution data objects (Detailed, non-digest form).

        Args:
          run: The tfdbg2 run to get `ExecutionDigest`s from.
          begin: Beginning execution index.
          end: Ending execution index.

        Returns:
          A JSON-serializable object containing the `ExecutionDigest`s and
          related meta-information
        """
    runs = self.Runs()
    if run not in runs:
        return None
    execution_digests = self._reader.executions(digest=True)
    end = self._checkBeginEndIndices(begin, end, len(execution_digests))
    execution_digests = execution_digests[begin:end]
    executions = self._reader.executions(digest=False, begin=begin, end=end)
    return {'begin': begin, 'end': end, 'executions': [execution.to_json() for execution in executions]}