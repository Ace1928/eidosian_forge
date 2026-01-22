import numbers
from torch.distributions import constraints, transforms
@biject_to.register(constraints.positive)
@biject_to.register(constraints.nonnegative)
@transform_to.register(constraints.positive)
@transform_to.register(constraints.nonnegative)
def _transform_to_positive(constraint):
    return transforms.ExpTransform()