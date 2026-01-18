from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from .util import pad_center, fill_off_diagonal, is_positive_int, tiny, expand_to
from .util.exceptions import ParameterError
from .filters import get_window
from typing import Any, Iterable, List, Optional, Tuple, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _IntLike_co
def viterbi_binary(prob: np.ndarray, transition: np.ndarray, *, p_state: Optional[np.ndarray]=None, p_init: Optional[np.ndarray]=None, return_logp: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Viterbi decoding from binary (multi-label), discriminative state predictions.

    Given a sequence of conditional state predictions ``prob[s, t]``,
    indicating the conditional likelihood of state ``s`` being active
    conditional on observation at time ``t``, and a 2*2 transition matrix
    ``transition`` which encodes the conditional probability of moving from
    state ``s`` to state ``~s`` (not-``s``), the Viterbi algorithm computes the
    most likely sequence of states from the observations.

    This function differs from `viterbi_discriminative` in that it does not assume the
    states to be mutually exclusive.  `viterbi_binary` is implemented by
    transforming the multi-label decoding problem to a collection
    of binary Viterbi problems (one for each *state* or label).

    The output is a binary matrix ``states[s, t]`` indicating whether each
    state ``s`` is active at time ``t``.

    Like `viterbi_discriminative`, the probabilities of the optimal state sequences
    are not normalized here.  If using the `return_logp=True` option (see below),
    be aware that the "probabilities" may not sum to (and may exceed) 1.

    Parameters
    ----------
    prob : np.ndarray [shape=(..., n_steps,) or (..., n_states, n_steps)], non-negative
        ``prob[s, t]`` is the probability of state ``s`` being active
        conditional on the observation at time ``t``.
        Must be non-negative and less than 1.

        If ``prob`` is 1-dimensional, it is expanded to shape ``(1, n_steps)``.

        If ``prob`` contains multiple input channels, then each channel is decoded independently.

    transition : np.ndarray [shape=(2, 2) or (n_states, 2, 2)], non-negative
        If 2-dimensional, the same transition matrix is applied to each sub-problem.
        ``transition[0, i]`` is the probability of the state going from inactive to ``i``,
        ``transition[1, i]`` is the probability of the state going from active to ``i``.
        Each row must sum to 1.

        If 3-dimensional, ``transition[s]`` is interpreted as the 2x2 transition matrix
        for state label ``s``.

    p_state : np.ndarray [shape=(n_states,)]
        Optional: marginal probability for each state (between [0,1]).
        If not provided, a uniform distribution (0.5 for each state)
        is assumed.

    p_init : np.ndarray [shape=(n_states,)]
        Optional: initial state distribution.
        If not provided, it is assumed to be uniform.

    return_logp : bool
        If ``True``, return the (unnormalized) log-likelihood of the state sequences.

    Returns
    -------
    Either ``states`` or ``(states, logp)``:
    states : np.ndarray [shape=(..., n_states, n_steps)]
        The most likely state sequence.
    logp : np.ndarray [shape=(..., n_states,)]
        If ``return_logp=True``, the (unnormalized) log probability of each
        state activation sequence ``states``

    See Also
    --------
    viterbi :
        Viterbi decoding from observation likelihoods
    viterbi_discriminative :
        Viterbi decoding for discriminative (mutually exclusive) state predictions

    Examples
    --------
    In this example, we have a sequence of binary state likelihoods that we want to de-noise
    under the assumption that state changes are relatively uncommon.  Positive predictions
    should only be retained if they persist for multiple steps, and any transient predictions
    should be considered as errors.  This use case arises frequently in problems such as
    instrument recognition, where state activations tend to be stable over time, but subject
    to abrupt changes (e.g., when an instrument joins the mix).

    We assume that the 0 state has a self-transition probability of 90%, and the 1 state
    has a self-transition probability of 70%.  We assume the marginal and initial
    probability of either state is 50%.

    >>> trans = np.array([[0.9, 0.1], [0.3, 0.7]])
    >>> prob = np.array([0.1, 0.7, 0.4, 0.3, 0.8, 0.9, 0.8, 0.2, 0.6, 0.3])
    >>> librosa.sequence.viterbi_binary(prob, trans, p_state=0.5, p_init=0.5)
    array([[0, 0, 0, 0, 1, 1, 1, 0, 0, 0]])
    """
    prob = np.atleast_2d(prob)
    n_states, n_steps = prob.shape[-2:]
    if transition.shape == (2, 2):
        transition = np.tile(transition, (n_states, 1, 1))
    elif transition.shape != (n_states, 2, 2):
        raise ParameterError(f'transition.shape={transition.shape}, must be (2, 2) or (n_states, 2, 2)={n_states}')
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=-1), 1):
        raise ParameterError('Invalid transition matrix: must be non-negative and sum to 1 on each row.')
    if np.any(prob < 0) or np.any(prob > 1):
        raise ParameterError('Invalid probability values: prob must be between [0, 1]')
    if p_state is None:
        p_state = np.empty(n_states)
        p_state.fill(0.5)
    else:
        p_state = np.atleast_1d(p_state)
    assert p_state is not None
    if p_state.shape != (n_states,) or np.any(p_state < 0) or np.any(p_state > 1):
        raise ParameterError(f'Invalid marginal state distributions: p_state={p_state}')
    if p_init is None:
        p_init = np.empty(n_states)
        p_init.fill(0.5)
    else:
        p_init = np.atleast_1d(p_init)
    assert p_init is not None
    if p_init.shape != (n_states,) or np.any(p_init < 0) or np.any(p_init > 1):
        raise ParameterError(f'Invalid initial state distributions: p_init={p_init}')
    shape_prefix = list(prob.shape[:-2])
    states = np.empty(shape_prefix + [n_states, n_steps], dtype=np.uint16)
    logp = np.empty(shape_prefix + [n_states])
    prob_binary = np.empty(shape_prefix + [2, n_steps])
    p_state_binary = np.empty(2)
    p_init_binary = np.empty(2)
    for state in range(n_states):
        prob_binary[..., 0, :] = 1 - prob[..., state, :]
        prob_binary[..., 1, :] = prob[..., state, :]
        p_state_binary[0] = 1 - p_state[state]
        p_state_binary[1] = p_state[state]
        p_init_binary[0] = 1 - p_init[state]
        p_init_binary[1] = p_init[state]
        states[..., state, :], logp[..., state] = viterbi_discriminative(prob_binary, transition[state], p_state=p_state_binary, p_init=p_init_binary, return_logp=True)
    if return_logp:
        return (states, logp)
    return states