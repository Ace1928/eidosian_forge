import numbers
from torch.distributions import constraints, transforms
@biject_to.register(constraints.stack)
def _biject_to_stack(constraint):
    return transforms.StackTransform([biject_to(c) for c in constraint.cseq], constraint.dim)