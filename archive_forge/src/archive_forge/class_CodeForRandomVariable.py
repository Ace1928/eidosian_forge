from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import reflection_using_prepare as rup
from cirq_ft.algos import select_and_prepare as sp
from cirq_ft.algos.mean_estimation import complex_phase_oracle
@attr.frozen
class CodeForRandomVariable:
    """A collection of `encoder` and `synthesizer` for a random variable y.

    We say we have "the code" for a random variable $y$ defined on a probability space
    $(W, p)$ if we have both, a synthesizer and an encoder defined as follows:

    The synthesizer is responsible to "prepare" the state
    $\\sum_{w \\in W} \\sqrt{p(w)} |w> |garbage_{w}>$ on the "selection register" $w$ and potentially
    using a "junk register" corresponding to $|garbage_{w}>$. Thus, for convenience, the synthesizer
    follows the LCU PREPARE Oracle API.
    $$
    synthesizer|0> = \\sum_{w \\in W} \\sqrt{p(w)} |w> |garbage_{w}>
    $$


    The encoder is responsible to encode the value of random variable $y(w)$ in a "target register"
    when the corresponding "selection register" stores integer $w$. Thus, for convenience, the
    encoder follows the LCU SELECT Oracle API.
    $$
    encoder|w>|0^b> = |w>|y(w)>
    $$
    where b is the number of bits required to encode the real range of random variable y.

    References:
        https://arxiv.org/abs/2208.07544, Definition 2.2 for synthesizer (P) and
        Definition 2.10 for encoder (Y).
    """
    synthesizer: sp.PrepareOracle
    encoder: sp.SelectOracle

    def __attrs_post_init__(self):
        assert self.synthesizer.selection_registers == self.encoder.selection_registers