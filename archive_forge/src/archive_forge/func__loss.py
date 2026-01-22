import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
def _loss(self, F, x1, x2):
    """
        the function computes the kl divergence between the negative distances
        (internally it compute a softmax casting into probabilities) and the
        identity matrix.

        This assumes that the two batches are aligned therefore the more similar
        vector should be the one having the same id.

        Batch1                                Batch2

        President of France                   French President
        President of US                       American President

        Given the question president of France in batch 1 the model will
        learn to predict french president comparing it with all the other
        vectors in batch 2
        """
    batch_size = x1.shape[0]
    labels = self._compute_labels(F, batch_size)
    distances = self._compute_distances(x1, x2)
    log_probabilities = F.log_softmax(-distances, axis=1)
    return self.kl_loss(log_probabilities, labels.as_in_context(distances.context))