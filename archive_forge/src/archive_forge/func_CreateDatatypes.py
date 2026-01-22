from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def CreateDatatypes(*ds):
    """Create mutually recursive Z3 datatypes using 1 or more Datatype helper objects.

    In the following example we define a Tree-List using two mutually recursive datatypes.

    >>> TreeList = Datatype('TreeList')
    >>> Tree     = Datatype('Tree')
    >>> # Tree has two constructors: leaf and node
    >>> Tree.declare('leaf', ('val', IntSort()))
    >>> # a node contains a list of trees
    >>> Tree.declare('node', ('children', TreeList))
    >>> TreeList.declare('nil')
    >>> TreeList.declare('cons', ('car', Tree), ('cdr', TreeList))
    >>> Tree, TreeList = CreateDatatypes(Tree, TreeList)
    >>> Tree.val(Tree.leaf(10))
    val(leaf(10))
    >>> simplify(Tree.val(Tree.leaf(10)))
    10
    >>> n1 = Tree.node(TreeList.cons(Tree.leaf(10), TreeList.cons(Tree.leaf(20), TreeList.nil)))
    >>> n1
    node(cons(leaf(10), cons(leaf(20), nil)))
    >>> n2 = Tree.node(TreeList.cons(n1, TreeList.nil))
    >>> simplify(n2 == n1)
    False
    >>> simplify(TreeList.car(Tree.children(n2)) == n1)
    True
    """
    ds = _get_args(ds)
    if z3_debug():
        _z3_assert(len(ds) > 0, 'At least one Datatype must be specified')
        _z3_assert(all([isinstance(d, Datatype) for d in ds]), 'Arguments must be Datatypes')
        _z3_assert(all([d.ctx == ds[0].ctx for d in ds]), 'Context mismatch')
        _z3_assert(all([d.constructors != [] for d in ds]), 'Non-empty Datatypes expected')
    ctx = ds[0].ctx
    num = len(ds)
    names = (Symbol * num)()
    out = (Sort * num)()
    clists = (ConstructorList * num)()
    to_delete = []
    for i in range(num):
        d = ds[i]
        names[i] = to_symbol(d.name, ctx)
        num_cs = len(d.constructors)
        cs = (Constructor * num_cs)()
        for j in range(num_cs):
            c = d.constructors[j]
            cname = to_symbol(c[0], ctx)
            rname = to_symbol(c[1], ctx)
            fs = c[2]
            num_fs = len(fs)
            fnames = (Symbol * num_fs)()
            sorts = (Sort * num_fs)()
            refs = (ctypes.c_uint * num_fs)()
            for k in range(num_fs):
                fname = fs[k][0]
                ftype = fs[k][1]
                fnames[k] = to_symbol(fname, ctx)
                if isinstance(ftype, Datatype):
                    if z3_debug():
                        _z3_assert(ds.count(ftype) == 1, 'One and only one occurrence of each datatype is expected')
                    sorts[k] = None
                    refs[k] = ds.index(ftype)
                else:
                    if z3_debug():
                        _z3_assert(is_sort(ftype), 'Z3 sort expected')
                    sorts[k] = ftype.ast
                    refs[k] = 0
            cs[j] = Z3_mk_constructor(ctx.ref(), cname, rname, num_fs, fnames, sorts, refs)
            to_delete.append(ScopedConstructor(cs[j], ctx))
        clists[i] = Z3_mk_constructor_list(ctx.ref(), num_cs, cs)
        to_delete.append(ScopedConstructorList(clists[i], ctx))
    Z3_mk_datatypes(ctx.ref(), num, names, out, clists)
    result = []
    for i in range(num):
        dref = DatatypeSortRef(out[i], ctx)
        num_cs = dref.num_constructors()
        for j in range(num_cs):
            cref = dref.constructor(j)
            cref_name = cref.name()
            cref_arity = cref.arity()
            if cref.arity() == 0:
                cref = cref()
            setattr(dref, cref_name, cref)
            rref = dref.recognizer(j)
            setattr(dref, 'is_' + cref_name, rref)
            for k in range(cref_arity):
                aref = dref.accessor(j, k)
                setattr(dref, aref.name(), aref)
        result.append(dref)
    return tuple(result)