import abc
import weakref
from numba.core import errors
class BasicRetarget(BaseRetarget):
    """A basic retargeting implementation for a single output target.

    This class has two abstract methods/properties that subclasses must define.

    - `output_target` must return output target name.
    - `compile_retarget` must define the logic to retarget the given dispatcher.

    By default, this class uses `RetargetCache` as the internal cache. This
    can be modified by overriding the `.cache_type` class attribute.

    """
    cache_type = RetargetCache

    def __init__(self):
        self.cache = self.cache_type()

    @abc.abstractproperty
    def output_target(self) -> str:
        """Returns the output target name.

        See numba/tests/test_retargeting.py for example usage.
        """
        pass

    @abc.abstractmethod
    def compile_retarget(self, orig_disp):
        """Returns the retargeted dispatcher.

        See numba/tests/test_retargeting.py for example usage.
        """
        pass

    def check_compatible(self, orig_disp):
        """
        This implementation checks that
        `self.output_target == orig_disp._required_target_backend`
        """
        required_target = orig_disp._required_target_backend
        output_target = self.output_target
        if required_target is not None:
            if output_target != required_target:
                m = f'The output target does match the required target: {output_target} != {required_target}.'
                raise errors.CompilerError(m)

    def retarget(self, orig_disp):
        """Apply retargeting to orig_disp.

        The retargeted dispatchers are cached for future use.
        """
        cache = self.cache
        opts = orig_disp.targetoptions
        if opts.get('target_backend') == self.output_target:
            return orig_disp
        cached = cache.load_cache(orig_disp)
        if cached is None:
            out = self.compile_retarget(orig_disp)
            cache.save_cache(orig_disp, out)
        else:
            out = cached
        return out