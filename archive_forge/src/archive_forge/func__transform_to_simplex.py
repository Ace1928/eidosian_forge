import numbers
from torch.distributions import constraints, transforms
@transform_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    return transforms.SoftmaxTransform()