from torch.fx.experimental.graph_gradual_typechecker import Refine
from torch.fx.tensor_type import TensorType
from torch.fx.experimental.unification import Var, unify  # type: ignore[attr-defined]

    A check equality to be used in fixed points.
    We do not use graph equality but instead type
    equality.
    