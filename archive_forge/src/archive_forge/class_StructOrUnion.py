import types
import weakref
from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing
class StructOrUnion(StructOrUnionOrEnum):
    fixedlayout = None
    completed = 0
    partial = False
    packed = 0

    def __init__(self, name, fldnames, fldtypes, fldbitsize, fldquals=None):
        self.name = name
        self.fldnames = fldnames
        self.fldtypes = fldtypes
        self.fldbitsize = fldbitsize
        self.fldquals = fldquals
        self.build_c_name_with_marker()

    def anonymous_struct_fields(self):
        if self.fldtypes is not None:
            for name, type in zip(self.fldnames, self.fldtypes):
                if name == '' and isinstance(type, StructOrUnion):
                    yield type

    def enumfields(self, expand_anonymous_struct_union=True):
        fldquals = self.fldquals
        if fldquals is None:
            fldquals = (0,) * len(self.fldnames)
        for name, type, bitsize, quals in zip(self.fldnames, self.fldtypes, self.fldbitsize, fldquals):
            if name == '' and isinstance(type, StructOrUnion) and expand_anonymous_struct_union:
                for result in type.enumfields():
                    yield result
            else:
                yield (name, type, bitsize, quals)

    def force_flatten(self):
        names = []
        types = []
        bitsizes = []
        fldquals = []
        for name, type, bitsize, quals in self.enumfields():
            names.append(name)
            types.append(type)
            bitsizes.append(bitsize)
            fldquals.append(quals)
        self.fldnames = tuple(names)
        self.fldtypes = tuple(types)
        self.fldbitsize = tuple(bitsizes)
        self.fldquals = tuple(fldquals)

    def get_cached_btype(self, ffi, finishlist, can_delay=False):
        BType = StructOrUnionOrEnum.get_cached_btype(self, ffi, finishlist, can_delay)
        if not can_delay:
            self.finish_backend_type(ffi, finishlist)
        return BType

    def finish_backend_type(self, ffi, finishlist):
        if self.completed:
            if self.completed != 2:
                raise NotImplementedError("recursive structure declaration for '%s'" % (self.name,))
            return
        BType = ffi._cached_btypes[self]
        self.completed = 1
        if self.fldtypes is None:
            pass
        elif self.fixedlayout is None:
            fldtypes = [tp.get_cached_btype(ffi, finishlist) for tp in self.fldtypes]
            lst = list(zip(self.fldnames, fldtypes, self.fldbitsize))
            extra_flags = ()
            if self.packed:
                if self.packed == 1:
                    extra_flags = (8,)
                else:
                    extra_flags = (0, self.packed)
            ffi._backend.complete_struct_or_union(BType, lst, self, -1, -1, *extra_flags)
        else:
            fldtypes = []
            fieldofs, fieldsize, totalsize, totalalignment = self.fixedlayout
            for i in range(len(self.fldnames)):
                fsize = fieldsize[i]
                ftype = self.fldtypes[i]
                if isinstance(ftype, ArrayType) and ftype.length_is_unknown():
                    BItemType = ftype.item.get_cached_btype(ffi, finishlist)
                    nlen, nrest = divmod(fsize, ffi.sizeof(BItemType))
                    if nrest != 0:
                        self._verification_error("field '%s.%s' has a bogus size?" % (self.name, self.fldnames[i] or '{}'))
                    ftype = ftype.resolve_length(nlen)
                    self.fldtypes = self.fldtypes[:i] + (ftype,) + self.fldtypes[i + 1:]
                BFieldType = ftype.get_cached_btype(ffi, finishlist)
                if isinstance(ftype, ArrayType) and ftype.length is None:
                    assert fsize == 0
                else:
                    bitemsize = ffi.sizeof(BFieldType)
                    if bitemsize != fsize:
                        self._verification_error("field '%s.%s' is declared as %d bytes, but is really %d bytes" % (self.name, self.fldnames[i] or '{}', bitemsize, fsize))
                fldtypes.append(BFieldType)
            lst = list(zip(self.fldnames, fldtypes, self.fldbitsize, fieldofs))
            ffi._backend.complete_struct_or_union(BType, lst, self, totalsize, totalalignment)
        self.completed = 2

    def _verification_error(self, msg):
        raise VerificationError(msg)

    def check_not_partial(self):
        if self.partial and self.fixedlayout is None:
            raise VerificationMissing(self._get_c_name())

    def build_backend_type(self, ffi, finishlist):
        self.check_not_partial()
        finishlist.append(self)
        return global_cache(self, ffi, 'new_%s_type' % self.kind, self.get_official_name(), key=self)