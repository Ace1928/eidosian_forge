import numbers
from torch.distributions import constraints, transforms
class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """

    def __init__(self):
        self._registry = {}
        super().__init__()

    def register(self, constraint, factory=None):
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (Callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        """
        if factory is None:
            return lambda factory: self.register(constraint, factory)
        if isinstance(constraint, constraints.Constraint):
            constraint = type(constraint)
        if not isinstance(constraint, type) or not issubclass(constraint, constraints.Constraint):
            raise TypeError(f'Expected constraint to be either a Constraint subclass or instance, but got {constraint}')
        self._registry[constraint] = factory
        return factory

    def __call__(self, constraint):
        """
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.arg_constraints['scale']
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)           # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        """
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(f'Cannot transform {type(constraint).__name__} constraints') from None
        return factory(constraint)