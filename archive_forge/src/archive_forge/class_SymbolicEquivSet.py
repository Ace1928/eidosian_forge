import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
class SymbolicEquivSet(ShapeEquivSet):
    """Just like ShapeEquivSet, except that it also reasons about variable
    equivalence symbolically by using their arithmetic definitions.
    The goal is to automatically derive the equivalence of array ranges
    (slicing). For instance, a[1:m] and a[0:m-1] shall be considered
    size-equivalence.
    """

    def __init__(self, typemap, def_by=None, ref_by=None, ext_shapes=None, defs=None, ind_to_var=None, obj_to_ind=None, ind_to_obj=None, next_id=0):
        """Create a new SymbolicEquivSet object, where typemap is a dictionary
        that maps variable names to their types, and it will not be modified.
        Optional keyword arguments are for internal use only.
        """
        self.def_by = def_by if def_by else {}
        self.ref_by = ref_by if ref_by else {}
        self.ext_shapes = ext_shapes if ext_shapes else {}
        self.rel_map = {}
        self.wrap_map = {}
        super(SymbolicEquivSet, self).__init__(typemap, defs, ind_to_var, obj_to_ind, ind_to_obj, next_id)

    def empty(self):
        """Return an empty SymbolicEquivSet.
        """
        return SymbolicEquivSet(self.typemap)

    def __repr__(self):
        return 'SymbolicEquivSet({}, ind_to_var={}, def_by={}, ref_by={}, ext_shapes={})'.format(self.ind_to_obj, self.ind_to_var, self.def_by, self.ref_by, self.ext_shapes)

    def clone(self):
        """Return a new copy.
        """
        return SymbolicEquivSet(self.typemap, def_by=copy.copy(self.def_by), ref_by=copy.copy(self.ref_by), ext_shapes=copy.copy(self.ext_shapes), defs=copy.copy(self.defs), ind_to_var=copy.copy(self.ind_to_var), obj_to_ind=copy.deepcopy(self.obj_to_ind), ind_to_obj=copy.deepcopy(self.ind_to_obj), next_id=self.next_ind)

    def get_rel(self, name):
        """Retrieve a definition pair for the given variable,
        or return None if it is not available.
        """
        return guard(self._get_or_set_rel, name)

    def _get_or_set_rel(self, name, func_ir=None):
        """Retrieve a definition pair for the given variable,
        and if it is not already available, try to look it up
        in the given func_ir, and remember it for future use.
        """
        if isinstance(name, ir.Var):
            name = name.name
        require(self.defs.get(name, 0) == 1)
        if name in self.def_by:
            return self.def_by[name]
        else:
            require(func_ir is not None)

            def plus(x, y):
                x_is_const = isinstance(x, int)
                y_is_const = isinstance(y, int)
                if x_is_const:
                    if y_is_const:
                        return x + y
                    else:
                        var, offset = y
                        return (var, x + offset)
                else:
                    var, offset = x
                    if y_is_const:
                        return (var, y + offset)
                    else:
                        return None

            def minus(x, y):
                if isinstance(y, int):
                    return plus(x, -y)
                elif isinstance(x, tuple) and isinstance(y, tuple) and (x[0] == y[0]):
                    return minus(x[1], y[1])
                else:
                    return None
            expr = get_definition(func_ir, name)
            value = (name, 0)
            if isinstance(expr, ir.Expr):
                if expr.op == 'call':
                    fname, mod_name = find_callname(func_ir, expr, typemap=self.typemap)
                    if fname == 'wrap_index' and mod_name == 'numba.parfors.array_analysis':
                        index = tuple((self.obj_to_ind.get(x.name, -1) for x in expr.args))
                        if -1 in index:
                            return None
                        names = self.ext_shapes.get(index, [])
                        names.append(name)
                        if len(names) > 0:
                            self._insert(names)
                        self.ext_shapes[index] = names
                elif expr.op == 'binop':
                    lhs = self._get_or_set_rel(expr.lhs, func_ir)
                    rhs = self._get_or_set_rel(expr.rhs, func_ir)
                    if lhs is None or rhs is None:
                        return None
                    elif expr.fn == operator.add:
                        value = plus(lhs, rhs)
                    elif expr.fn == operator.sub:
                        value = minus(lhs, rhs)
            elif isinstance(expr, ir.Const) and isinstance(expr.value, int):
                value = expr.value
            require(value is not None)
            self.def_by[name] = value
            if isinstance(value, int) or (isinstance(value, tuple) and (value[0] != name or value[1] != 0)):
                if isinstance(value, tuple):
                    var, offset = value
                    if not var in self.ref_by:
                        self.ref_by[var] = []
                    self.ref_by[var].append((name, -offset))
                    ind = self._get_ind(var)
                    if ind >= 0:
                        objs = self.ind_to_obj[ind]
                        names = []
                        for obj in objs:
                            if obj in self.ref_by:
                                names += [x for x, i in self.ref_by[obj] if i == -offset]
                        if len(names) > 1:
                            super(SymbolicEquivSet, self)._insert(names)
            return value

    def define(self, var, redefined, func_ir=None, typ=None):
        """Besides incrementing the definition count of the given variable
        name, it will also retrieve and simplify its definition from func_ir,
        and remember the result for later equivalence comparison. Supported
        operations are:
          1. arithmetic plus and minus with constants
          2. wrap_index (relative to some given size)
        """
        if isinstance(var, ir.Var):
            name = var.name
        else:
            name = var
        super(SymbolicEquivSet, self).define(name, redefined)
        if func_ir and self.defs.get(name, 0) == 1 and isinstance(typ, types.Number):
            value = guard(self._get_or_set_rel, name, func_ir)
            if isinstance(value, int):
                self._insert([name, value])
            if isinstance(var, ir.Var):
                ind = self._get_or_add_ind(name)
                if not ind in self.ind_to_obj:
                    self.ind_to_obj[ind] = [name]
                    self.obj_to_ind[name] = ind
                if ind in self.ind_to_var:
                    self.ind_to_var[ind].append(var)
                else:
                    self.ind_to_var[ind] = [var]
            return True

    def _insert(self, objs):
        """Overload _insert method to handle ind changes between relative
        objects.  Returns True if some change is made, false otherwise.
        """
        indset = set()
        uniqs = set()
        for obj in objs:
            ind = self._get_ind(obj)
            if ind == -1:
                uniqs.add(obj)
            elif not ind in indset:
                uniqs.add(obj)
                indset.add(ind)
        if len(uniqs) <= 1:
            return False
        uniqs = list(uniqs)
        super(SymbolicEquivSet, self)._insert(uniqs)
        objs = self.ind_to_obj[self._get_ind(uniqs[0])]
        offset_dict = {}

        def get_or_set(d, k):
            if k in d:
                v = d[k]
            else:
                v = []
                d[k] = v
            return v
        for obj in objs:
            if obj in self.def_by:
                value = self.def_by[obj]
                if isinstance(value, tuple):
                    name, offset = value
                    get_or_set(offset_dict, -offset).append(name)
                    if name in self.ref_by:
                        for v, i in self.ref_by[name]:
                            get_or_set(offset_dict, -(offset + i)).append(v)
            if obj in self.ref_by:
                for name, offset in self.ref_by[obj]:
                    get_or_set(offset_dict, offset).append(name)
        for names in offset_dict.values():
            self._insert(names)
        return True

    def set_shape_setitem(self, obj, shape):
        """remember shapes of SetItem IR nodes.
        """
        assert isinstance(obj, (ir.StaticSetItem, ir.SetItem))
        self.ext_shapes[obj] = shape

    def _get_shape(self, obj):
        """Overload _get_shape to retrieve the shape of SetItem IR nodes.
        """
        if isinstance(obj, (ir.StaticSetItem, ir.SetItem)):
            require(obj in self.ext_shapes)
            return self.ext_shapes[obj]
        else:
            assert isinstance(obj, ir.Var)
            typ = self.typemap[obj.name]
            if isinstance(typ, types.SliceType):
                return (obj,)
            else:
                return super(SymbolicEquivSet, self)._get_shape(obj)