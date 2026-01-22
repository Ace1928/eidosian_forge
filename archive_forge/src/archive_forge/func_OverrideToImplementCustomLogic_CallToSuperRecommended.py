from ray.rllib.utils.deprecation import Deprecated
from ray.util.annotations import _mark_annotated
def OverrideToImplementCustomLogic_CallToSuperRecommended(obj):
    """Users should override this in their sub-classes to implement custom logic.

    Thereby, it is recommended (but not required) to call the super-class'
    corresponding method.

    Used in Algorithm and Policy to tag methods that need overriding, but the
    super class' method should still be called, e.g.
    `Algorithm.setup()`.

    .. testcode::
        :skipif: True

        from ray import tune
        @overrides(tune.Trainable)
        @OverrideToImplementCustomLogic_CallToSuperRecommended
        def setup(self, config):
            # implement custom setup logic here ...
            super().setup(config)
            # ... or here (after having called super()'s setup method.
    """
    obj.__is_overriden__ = False
    return obj