import numbers
from torch.distributions import constraints, transforms
@transform_to.register(constraints.lower_cholesky)
def _transform_to_lower_cholesky(constraint):
    return transforms.LowerCholeskyTransform()