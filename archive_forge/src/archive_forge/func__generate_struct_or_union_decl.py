import sys, os
import types
from . import model
from .error import VerificationError
def _generate_struct_or_union_decl(self, tp, prefix, name):
    if tp.fldnames is None:
        return
    checkfuncname = '_cffi_check_%s_%s' % (prefix, name)
    layoutfuncname = '_cffi_layout_%s_%s' % (prefix, name)
    cname = ('%s %s' % (prefix, name)).strip()
    prnt = self._prnt
    prnt('static void %s(%s *p)' % (checkfuncname, cname))
    prnt('{')
    prnt('  /* only to generate compile-time warnings or errors */')
    prnt('  (void)p;')
    for fname, ftype, fbitsize, fqual in tp.enumfields():
        if isinstance(ftype, model.PrimitiveType) and ftype.is_integer_type() or fbitsize >= 0:
            prnt('  (void)((p->%s) << 1);' % fname)
        else:
            try:
                prnt('  { %s = &p->%s; (void)tmp; }' % (ftype.get_c_name('*tmp', 'field %r' % fname, quals=fqual), fname))
            except VerificationError as e:
                prnt('  /* %s */' % str(e))
    prnt('}')
    self.export_symbols.append(layoutfuncname)
    prnt('intptr_t %s(intptr_t i)' % (layoutfuncname,))
    prnt('{')
    prnt('  struct _cffi_aligncheck { char x; %s y; };' % cname)
    prnt('  static intptr_t nums[] = {')
    prnt('    sizeof(%s),' % cname)
    prnt('    offsetof(struct _cffi_aligncheck, y),')
    for fname, ftype, fbitsize, fqual in tp.enumfields():
        if fbitsize >= 0:
            continue
        prnt('    offsetof(%s, %s),' % (cname, fname))
        if isinstance(ftype, model.ArrayType) and ftype.length is None:
            prnt('    0,  /* %s */' % ftype._get_c_name())
        else:
            prnt('    sizeof(((%s *)0)->%s),' % (cname, fname))
    prnt('    -1')
    prnt('  };')
    prnt('  return nums[i];')
    prnt('  /* the next line is not executed, but compiled */')
    prnt('  %s(0);' % (checkfuncname,))
    prnt('}')
    prnt()