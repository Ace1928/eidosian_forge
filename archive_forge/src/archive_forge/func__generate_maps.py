from pyomo.contrib.pynumero.interfaces.nlp import NLP, ExtendedNLP
import numpy as np
import scipy.sparse as sp
def _generate_maps(self):
    if self._original_idxs is None or self._projected_idxs is None:
        primals_ordering_dict = {k: i for i, k in enumerate(self._primals_ordering)}
        original_names = self._original_nlp.primals_names()
        original_idxs = list()
        projected_idxs = list()
        for i, nm in enumerate(original_names):
            if nm in primals_ordering_dict:
                original_idxs.append(i)
                projected_idxs.append(primals_ordering_dict[nm])
        self._original_idxs = np.asarray(original_idxs)
        self._projected_idxs = np.asarray(projected_idxs)
        self._original_to_projected = np.nan * np.zeros(self._original_nlp.n_primals())
        self._original_to_projected[self._original_idxs] = self._projected_idxs