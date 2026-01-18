from tensorboard.plugins.hparams import metrics
from tensorboard.plugins.scalar import scalars_plugin
Executes the request.

        Returns:
            An array of tuples representing the metric evaluations--each of the
            form (<wall time in secs>, <training step>, <metric value>).
        