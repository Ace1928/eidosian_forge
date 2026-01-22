from ... import base as nib
from ..traits_extension import rebase_path_traits, resolve_path_traits, Path
class _test_spec(nib.TraitedSpec):
    a = nib.File()
    b = nib.traits.Tuple(nib.File(), nib.File())
    c = nib.traits.List(nib.File())
    d = nib.traits.Either(nib.File(), nib.traits.Float())
    e = nib.OutputMultiObject(nib.File())
    ee = nib.OutputMultiObject(nib.Str)
    f = nib.traits.Dict(nib.Str, nib.File())
    g = nib.traits.Either(nib.File, nib.Str)
    h = nib.Str
    i = nib.traits.Either(nib.File, nib.traits.Tuple(nib.File, nib.traits.Int))
    j = nib.traits.Either(nib.File, nib.traits.Tuple(nib.File, nib.traits.Int), nib.traits.Dict(nib.Str, nib.File()))
    k = nib.DictStrStr