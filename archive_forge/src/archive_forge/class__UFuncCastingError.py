from .._utils import set_module
@_display_as_base
class _UFuncCastingError(UFuncTypeError):

    def __init__(self, ufunc, casting, from_, to):
        super().__init__(ufunc)
        self.casting = casting
        self.from_ = from_
        self.to = to